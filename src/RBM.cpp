#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <sstream>
#include "utils.h"
#include "RBM.h"
using namespace std;
using namespace utils;

/*
  nv: number of visible nodes
  nh: number of hidden nodes
  nl: number of values in each node
  nk: CD-k
*/
RBM::RBM(int nv, int nh, int nl, int nk) {
  N_vis = nv;
  N_hid = nh;
  N_val = nl;

  float wm = 0.1 * sqrt(6. / (nh + nv * nl));
  W = new float**[N_vis];
  for (int i=0; i<N_vis; i++) {
    W[i] = new float*[N_hid];
    for (int j=0; j<N_hid; j++) {
      W[i][j] = new float[N_val];
      for (int k=0; k<N_val; k++) {
        W[i][j][k] = uniform(-wm, wm);
      }
    }
  }

  vb = new float*[N_vis];
  for (int i=0; i<N_vis; i++) {
    vb[i] = new float[N_val];
    for (int j=0; j<N_val; j++) {
      vb[i][j] = 0.0;
    }
  }

  hb = new float[N_hid];
  for (int i=0; i<N_hid; i++) {
    hb[i] = 0.0;
  }
}

RBM::~RBM() {
  for (int i=0; i<N_vis; i++) {
    for (int j=0; j<N_hid; j++) {
      delete [] W[i][j];
    }
    delete [] W[i];
  }
  delete [] W;

  for (int i=0; i<N_vis; i++) {
    delete [] vb[i];
  }
  delete [] vb;
  delete [] hb;
}

void RBM::train(int ***dat, int ndat, int ***test, int ntest, int epoch, float learning_rate=0.1) {
  //alloc matrix
  float **pos_hidden_activations = new float*[ndat];
  float **neg_hidden_activations = new float*[ndat];
  float **pos_hidden_probs = new float*[ndat];
  float **neg_hidden_probs = new float*[ndat];
  int **pos_hidden_states = new int*[ndat];
  float ***neg_visible_activations = new float**[ndat];
  float ***neg_visible_probs = new float**[ndat];
  int ***neg_visible_states = new int**[ndat];
  float *p = new float[N_val];

  for (int i=0; i<ndat; i++) {
    pos_hidden_activations[i] = new float[N_hid];
    neg_hidden_activations[i] = new float[N_hid];
    pos_hidden_probs[i] = new float[N_hid];
    neg_hidden_probs[i] = new float[N_hid];
    pos_hidden_states[i] = new int[N_hid];
    neg_visible_activations[i] = new float*[N_vis];
    neg_visible_probs[i] = new float*[N_vis];
    neg_visible_states[i] = new int*[N_vis];
    for (int j=0; j<N_vis; j++) {
      neg_visible_activations[i][j] = new float[N_val];
      neg_visible_probs[i][j] = new float[N_val];
      neg_visible_states[i][j] = new int[N_val];
    }
  }
  float ***pos_associations = new float**[N_vis];
  float **pos_associations_vb = new float*[N_vis];
  float *pos_associations_hb = new float[N_hid];
  float ***neg_associations = new float**[N_vis];
  float **neg_associations_vb = new float*[N_vis];
  float *neg_associations_hb = new float[N_hid];
  for (int i=0; i<N_vis; i++) {
    pos_associations[i] = new float*[N_hid];
    pos_associations_vb[i] = new float[N_val];
    neg_associations[i] = new float*[N_hid];
    neg_associations_vb[i] = new float[N_val];
    for (int j=0; j<N_hid; j++) {
      pos_associations[i][j] = new float[N_val];
      neg_associations[i][j] = new float[N_val];
    }
  }

  //GO!
  for (int n=0; n<=epoch; n++) {
    //pos_hidden_activations = np.dot(data, self.weights)
    for (int i=0; i<ndat; i++) {
      for (int j=0; j<N_hid; j++) {
        pos_hidden_activations[i][j] = 0.0;
        for (int k=0; k<N_vis; k++) {
          for (int l=0; l<N_val; l++) {
            pos_hidden_activations[i][j] += float(dat[i][k][l]) * W[k][j][l];
          }
        }
        pos_hidden_activations[i][j] += hb[j];
      }
    }

    //pos_hidden_probs = self._logistic(pos_hidden_activations)
    //pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
    for (int i=0; i<ndat; i++) {
      for (int j=0; j<N_hid; j++){
        pos_hidden_probs[i][j] = sigmoid(pos_hidden_activations[i][j]);
        pos_hidden_states[i][j] = int(pos_hidden_probs[i][j] > uniform(0,1));
      }
    }

    //pos_associations = np.dot(data.T, pos_hidden_probs)
    for (int i=0; i<N_vis; i++) {
      for (int l=0; l<N_val; l++) {
        for (int j=0; j<N_hid; j++) {
          pos_associations[i][j][l] = 0.0;
          for (int k=0; k<ndat; k++) {
            pos_associations[i][j][l] += float(dat[k][i][l]) * pos_hidden_probs[k][j];
          }
        }
        pos_associations_vb[i][l] = 0.0;
        for (int k=0; k<ndat; k++) {
          pos_associations_vb[i][l] += float(dat[k][i][l]);
        }
      }
    }
    for (int i=0; i<N_hid; i++) {
      pos_associations_hb[i] = 0.0;
      for (int j=0; j<ndat; j++) {
        pos_associations_hb[i] += pos_hidden_probs[j][i];
      }
    }

    //neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
    for (int i=0; i<ndat; i++) {
      for (int j=0; j<N_vis; j++) {
        for (int l=0; l<N_val; l++) {
          neg_visible_activations[i][j][l] = 0.0;
          for (int k=0; k<N_hid; k++) {
            neg_visible_activations[i][j][l] += float(pos_hidden_states[i][k]) * W[j][k][l];
          }
          neg_visible_activations[i][j][l] += vb[j][l];
        }
      }
    }

    //neg_visible_probs = self._logistic(neg_visible_activations) => softmax
    for (int i=0; i<ndat; i++) {
      for (int j=0; j<N_vis; j++) {
        float sum = 0.0;
        for (int l=0; l<N_val; l++) {
          neg_visible_probs[i][j][l] = exp(neg_visible_activations[i][j][l]);
          sum += neg_visible_probs[i][j][l];
        }
        for (int l=0; l<N_val; l++) {
          if (sum>0) {
            neg_visible_probs[i][j][l] /= sum;
          }
          else {
            neg_visible_probs[i][j][l] = 1.0/N_val;
          }
        }
      }
      //softmax
      for (int j=0; j<N_vis; j++) {
        // init p(i)
        double sum_p = 0.0;
        for (int l=0; l<N_val; l++) {
          neg_visible_states[i][j][l] = 0;
          sum_p += neg_visible_probs[i][j][l];
          p[l] = sum_p;
        }
        float ran = uniform(0,1);
        //cout << "ran=" << ran << endl;
        for (int l=0; l<N_val; l++) {
          if (ran<=p[l]) {
            neg_visible_states[i][j][l] = 1;
            break;
          }
        }
      }
    }

    //neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
    for (int i=0; i<ndat; i++) {
      for (int j=0; j<N_hid; j++) {
        neg_hidden_activations[i][j] = 0.0;
        for (int k=0; k<N_vis; k++) {
          for (int l=0; l<N_val; l++) {
            neg_hidden_activations[i][j] += float(neg_visible_states[i][k][l]) * W[k][j][l];
          }
        }
        neg_hidden_activations[i][j] += hb[j];
      }
    }

    //neg_hidden_probs = self._logistic(neg_hidden_activations)
    for (int i=0; i<ndat; i++) {
      for (int j=0; j<N_hid; j++) {
        neg_hidden_probs[i][j] = sigmoid(neg_hidden_activations[i][j]);
      }
    }

    //neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)
    for (int i=0; i<N_vis; i++) {
      for (int l=0; l<N_val; l++) {
        for (int j=0; j<N_hid; j++) {
          neg_associations[i][j][l] = 0.0;
          for (int k=0; k<ndat; k++) {
            neg_associations[i][j][l] += float(neg_visible_states[k][i][l]) * neg_hidden_probs[k][j];
          }
        }
        neg_associations_vb[i][l] = 0.0;
        for (int k=0; k<ndat; k++) {
          neg_associations_vb[i][l] += float(neg_visible_states[k][i][l]);
        }
      }
    }
    for (int i=0; i<N_hid; i++) {
      neg_associations_hb[i] = 0.0;
      for (int j=0; j<ndat; j++) {
        neg_associations_hb[i] += neg_hidden_probs[j][i];
      }
    }

    //update: currently fix the boundary of W
    //float max_w = 25.0;
    //float min_w = -25.0;
    //self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)
    for (int i=0; i<N_vis; i++) {
      for (int l=0; l<N_val; l++) {
        vb[i][l] += learning_rate * (pos_associations_vb[i][l]-neg_associations_vb[i][l]) / ndat;
        //if (vb[i][l]>max_w) vb[i][l] = max_w;
        //if (vb[i][l]<min_w) vb[i][l] = min_w;
        for (int j=0; j<N_hid; j++) {
          W[i][j][l] += learning_rate * (pos_associations[i][j][l]-neg_associations[i][j][l]) / ndat;
          //if (W[i][j][l]>max_w) W[i][j][l] = max_w;
          //if (W[i][j][l]<min_w) W[i][j][l] = min_w;
        }
      }
    }
    for (int i=0; i<N_hid; i++) {
      hb[i] += learning_rate * (pos_associations_hb[i]-neg_associations_hb[i]) / ndat;
      //if (hb[i]>max_w) hb[i]=max_w;
      //if (hb[i]<min_w) hb[i]=min_w;
      //cout << hb[i] << ",";
    }
    //cout << endl;

    if (n%10 == 0) {
      cout << "epoch= " << n;
      check_status(dat, ndat, test, ntest);
    }
  }

  //free matrix
  for (int i=0; i<ndat; i++) {
    delete [] pos_hidden_activations[i];
    delete [] neg_hidden_activations[i];
    delete [] pos_hidden_probs[i];
    delete [] neg_hidden_probs[i];
    delete [] pos_hidden_states[i];
    for (int j=0; j<N_vis; j++) {
      delete [] neg_visible_activations[i][j];
      delete [] neg_visible_states[i][j];
      delete [] neg_visible_probs[i][j];
    }
    delete [] neg_visible_activations[i];
    delete [] neg_visible_states[i];
    delete [] neg_visible_probs[i];
  }
  delete [] pos_hidden_activations;
  delete [] pos_hidden_probs;
  delete [] pos_hidden_states;
  delete [] neg_visible_activations;
  delete [] neg_visible_states;
  delete [] neg_visible_probs;
  delete [] neg_hidden_probs;
  delete [] neg_hidden_activations;

  for (int i=0; i<N_vis; i++) {
    for (int j=0; j<N_hid; j++) {
      delete [] pos_associations[i][j];
      delete [] neg_associations[i][j];
    }
    delete [] pos_associations[i];
    delete [] pos_associations_vb[i];
    delete [] neg_associations[i];
    delete [] neg_associations_vb[i];
  }
  delete [] pos_associations;
  delete [] pos_associations_vb;
  delete [] pos_associations_hb;
  delete [] neg_associations;
  delete [] neg_associations_vb;
  delete [] neg_associations_hb;
  delete [] p;
}

void RBM::load(ifstream &fp) {
  //
  int nv, nh, nl;
  fp >> nv >> nh >> nl;
  if (N_vis != nv) {
    cerr << "N_vis mismatch in weights" << endl;
    exit(1);
  }
  if (N_hid != nh) {
    cerr << "N_hid mismatch in weights" << endl;
    exit(1);
  }
  if (N_val != nl) {
    cerr << "N_val mismatch in weights" << endl;
    exit(1);
  }
  //vb
  for (int i=0; i<N_vis; i++) {
    for (int l=0; l<N_val; l++) {
      fp >> vb[i][l];
    }
  }
  //hb
  for (int i=0; i<N_hid; i++) {
    fp >> hb[i];
  }
  //W
  for (int i=0; i<N_vis; i++) {
    for (int j=0; j<N_hid; j++) {
      for (int k=0; k<N_val; k++) {
        fp >> W[i][j][k];
      }
    }
  }
}

void RBM::save(ofstream &fp) {
  fp << N_vis << " " << N_hid << " " << N_val << endl;
  //vb
  for (int i=0; i<N_vis; i++) {
    for (int l=0; l<N_val; l++) {
      fp << vb[i][l] << " ";
    }
    fp << endl;
  }
  //hb
  for (int i=0; i<N_hid; i++) {
    fp << hb[i] << " ";
  }
  fp << endl;
  //W
  for (int i=0; i<N_vis; i++) {
    for (int j=0; j<N_hid; j++) {
      for (int k=0; k<N_val; k++) {
        fp << W[i][j][k] << " ";
      }
    }
    fp << endl;
  }
}

void RBM::daydream(int ndat) {
  // alloc
  int **v = new int*[N_vis];
  int *h = new int[N_hid];
  float **visible_activations = new float*[N_vis];
  float *hidden_activations = new float[N_hid];
  float **visible_probs = new float*[N_vis];
  float *hidden_probs = new float[N_hid];
  float *p = new float[N_val];
  for (int i=0; i<N_vis; i++) {
    v[i] = new int[N_val];
    visible_activations[i] = new float[N_val];
    visible_probs[i] = new float[N_val];
  }

  // random h
  for (int i=0; i<N_hid; i++) {
    h[i] = uniformint(0, 2);
  }

  // generate
  for (int n=0; n<=ndat; n++) {
    // h->v
    for (int i=0; i<N_vis; i++) {
      for (int l=0; l<N_val; l++) {
        visible_activations[i][l] = 0.0;
        for (int j=0; j<N_hid; j++) {
          visible_activations[i][l] += W[i][j][l] * h[j];
        }
        visible_activations[i][l] += vb[i][l];
      }
    }
    for (int i=0; i<N_vis; i++) {
      float sum = 0.0;
      for (int l=0; l<N_val; l++) {
        visible_probs[i][l] = sigmoid(visible_activations[i][l]);
        sum += visible_probs[i][l];
      }
      for (int l=0; l<N_val; l++) {
        visible_probs[i][l] /= sum;
      }
    }
    for (int i=0; i<N_vis; i++) {
      // init p(i)
      double sum_p = 0.0;
      for (int l=0; l<N_val; l++) {
        v[i][l] = 0;
        sum_p += visible_probs[i][l];
        p[l] = sum_p;
      }
      float ran = uniform(0,1);
      //cout << "ran=" << ran << endl;
      for (int l=0; l<N_val; l++) {
        if (ran<=p[l]) {
          cout << l;
          v[i][l] = 1;
          break;
        }
      }
    }
    cout << endl;
    // v->h
    for (int i=0; i<N_hid; i++) {
      hidden_activations[i] = 0.0;
      for (int j=0; j<N_vis; j++) {
        for (int l=0; l<N_val; l++) {
          hidden_activations[i] += W[j][i][l] * v[j][l];
        }
      }
      hidden_activations[i] += hb[i];
    }
    for (int i=0; i<N_hid; i++) {
      hidden_probs[i] = sigmoid(hidden_activations[i]);
      h[i] = int( hidden_probs[i] > uniform(0,1) );
    }
  }

  // delete
  for (int i=0; i<N_vis; i++) {
    delete [] v[i];
    delete [] visible_activations[i];
    delete [] visible_probs[i];
  }
  delete [] v;
  delete [] h;
  delete [] p;
  delete [] visible_activations;
  delete [] visible_probs;
  delete [] hidden_activations;
  delete [] hidden_probs;
}

void RBM::check_status(int ***dat, int ndat, int ***test, int ntest) {
  float **pos_hidden_activations = new float*[ndat+ntest];
  float **neg_hidden_activations = new float*[ndat];
  float **pos_hidden_probs = new float*[ndat];
  int **pos_hidden_states = new int*[ndat];
  float ***neg_visible_activations = new float**[ndat];
  float ***neg_visible_activations_p = new float**[ndat];
  float ***neg_visible_probs = new float**[ndat];
  float ***neg_visible_probs_p = new float**[ndat];
  int ***neg_visible_states = new int**[ndat];
  float *p = new float[N_val];
  float *vt = new float[ndat+ntest];
  float *ht = new float[ndat+ntest];

  for (int i=0; i<ndat; i++) {
    pos_hidden_activations[i] = new float[N_hid];
    neg_hidden_activations[i] = new float[N_hid];
    pos_hidden_probs[i] = new float[N_hid];
    pos_hidden_states[i] = new int[N_hid];
    neg_visible_activations[i] = new float*[N_vis];
    neg_visible_activations_p[i] = new float*[N_vis];
    neg_visible_probs[i] = new float*[N_vis];
    neg_visible_probs_p[i] = new float*[N_vis];
    neg_visible_states[i] = new int*[N_vis];
    for (int j=0; j<N_vis; j++) {
      neg_visible_activations[i][j] = new float[N_val];
      neg_visible_activations_p[i][j] = new float[N_val];
      neg_visible_probs[i][j] = new float[N_val];
      neg_visible_probs_p[i][j] = new float[N_val];
      neg_visible_states[i][j] = new int[N_val];
    }
  }
  for (int i=0; i<ntest; i++) {
    pos_hidden_activations[i+ndat] = new float[N_hid];
  }

  //pos_hidden_activations = np.dot(data, self.weights)
  for (int i=0; i<ndat+ntest; i++) {
    for (int j=0; j<N_hid; j++) {
      pos_hidden_activations[i][j] = 0.0;
      for (int k=0; k<N_vis; k++) {
        for (int l=0; l<N_val; l++) {
          if (i<ndat) {
            pos_hidden_activations[i][j] += float(dat[i][k][l]) * W[k][j][l];
          }
          else {
            pos_hidden_activations[i][j] += float(test[i-ndat][k][l]) * W[k][j][l];
          }
        }
      }
      pos_hidden_activations[i][j] += hb[j];
    }
    //vt
    vt[i] = 0.0;
    for (int j=0; j<N_vis; j++) {
      for (int l=0; l<N_val; l++) {
        if (i<ndat) {
          vt[i] += float(dat[i][j][l]) * vb[j][l];
        }
        else {
          vt[i] += float(test[i-ndat][j][l]) * vb[j][l];
        }
      }
    }
  }

  //ht
  for (int i=0; i<ndat+ntest; i++) {
    ht[i] = 0.0;
    for (int j=0; j<N_hid; j++) {
      ht[i] -= log(sigmoid(-pos_hidden_activations[i][j]));
    }
  }

  //pos_hidden_probs = self._logistic(pos_hidden_activations)
  //pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
  for (int i=0; i<ndat; i++) {
    for (int j=0; j<N_hid; j++){
      pos_hidden_probs[i][j] = sigmoid(pos_hidden_activations[i][j]);
      pos_hidden_states[i][j] = int(pos_hidden_probs[i][j] > uniform(0,1));
    }
  }

  //neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
  for (int i=0; i<ndat; i++) {
    for (int j=0; j<N_vis; j++) {
      for (int l=0; l<N_val; l++) {
        neg_visible_activations[i][j][l] = 0.0;
        neg_visible_activations_p[i][j][l] = 0.0;
        for (int k=0; k<N_hid; k++) {
          neg_visible_activations[i][j][l] += float(pos_hidden_states[i][k]) * W[j][k][l];
          neg_visible_activations_p[i][j][l] += pos_hidden_probs[i][k] * W[j][k][l];
        }
        neg_visible_activations[i][j][l] += vb[j][l];
        neg_visible_activations_p[i][j][l] += vb[j][l];
      }
    }
  }

  //neg_visible_probs = self._logistic(neg_visible_activations) => softmax
  for (int i=0; i<ndat; i++) {
    for (int j=0; j<N_vis; j++) {
      float sum = 0.0;
      float sum_p = 0.0;
      for (int l=0; l<N_val; l++) {
        neg_visible_probs[i][j][l] = exp(neg_visible_activations[i][j][l]);
        neg_visible_probs_p[i][j][l] = exp(neg_visible_activations_p[i][j][l]);
        sum += neg_visible_probs[i][j][l];
        sum_p += neg_visible_probs_p[i][j][l];
      }
      for (int l=0; l<N_val; l++) {
        neg_visible_probs[i][j][l] /= sum;
        neg_visible_probs_p[i][j][l] /= sum_p;
      }
    }
    //softmax
    for (int j=0; j<N_vis; j++) {
      // init p(i)
      double sum_p = 0.0;
      for (int l=0; l<N_val; l++) {
        neg_visible_states[i][j][l] = 0;
        sum_p += neg_visible_probs[i][j][l];
        p[l] = sum_p;
      }
      float ran = uniform(0,1);
      //cout << "ran=" << ran << endl;
      for (int l=0; l<N_val; l++) {
        if (ran<=p[l]) {
          neg_visible_states[i][j][l] = 1;
          break;
        }
      }
    }
  }

  //neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
  for (int i=0; i<ndat; i++) {
    for (int j=0; j<N_hid; j++) {
      neg_hidden_activations[i][j] = 0.0;
      for (int k=0; k<N_vis; k++) {
        for (int l=0; l<N_val; l++) {
          neg_hidden_activations[i][j] += float(neg_visible_states[i][k][l]) * W[k][j][l];
        }
      }
      neg_hidden_activations[i][j] += hb[j];
    }
  }

  float error = 0.0;
  float cross = 0.0;
  float f0 = 0.0;
  float f1 = 0.0;

  for (int i=0; i<ndat; i++) {
    for (int j=0; j<N_vis; j++) {
      for (int l=0; l<N_val; l++) {
        float d = float(dat[i][j][l]) - neg_visible_probs[i][j][l];
        error += d * d;
        cross += float(dat[i][j][l])*log(neg_visible_probs_p[i][j][l]);
      }
    }
    f0 -= vt[i];
    f0 -= ht[i];
  }
  for (int i=0; i<ntest; i++) {
    f1 -= vt[ndat+i];
    f1 -= ht[ndat+i];
  }
  f0 /= ndat;
  f1 /= ntest;

  cout << " error= " << error/ndat << " cross= " << -cross/ndat
    << " F_train= " << f0 << " F_test= " << f1 << endl;

  for (int i=0; i<ndat; i++) {
    delete [] pos_hidden_activations[i];
    delete [] neg_hidden_activations[i];
    delete [] pos_hidden_probs[i];
    delete [] pos_hidden_states[i];
    for (int j=0; j<N_vis; j++) {
      delete [] neg_visible_activations[i][j];
      delete [] neg_visible_activations_p[i][j];
      delete [] neg_visible_states[i][j];
      delete [] neg_visible_probs[i][j];
      delete [] neg_visible_probs_p[i][j];
    }
    delete [] neg_visible_activations[i];
    delete [] neg_visible_activations_p[i];
    delete [] neg_visible_states[i];
    delete [] neg_visible_probs[i];
    delete [] neg_visible_probs_p[i];
  }
  for (int i=0; i<ntest; i++) {
    delete [] pos_hidden_activations[i+ndat];
  }
  delete [] pos_hidden_activations;
  delete [] pos_hidden_probs;
  delete [] pos_hidden_states;
  delete [] neg_visible_activations;
  delete [] neg_visible_activations_p;
  delete [] neg_visible_states;
  delete [] neg_visible_probs;
  delete [] neg_visible_probs_p;
  delete [] neg_hidden_activations;

  delete [] p;
  delete [] vt;
  delete [] ht;
}

// rbm.x [nvis] [nhid] [nlevel] [ndat/datfile] [weights] [epochs] [nk] [lr]
int main(int argc, char *argv[]) {
  int nvis = atoi(argv[1]);
  int nhid = atoi(argv[2]);
  int nval = atoi(argv[3]);
  int ndat = 0;
  int ntest = 0;
  // load data from file if this is not a number
  if (isdigit(argv[4][0])) {
    ndat = atoi(argv[4]);
  }

  float lr = 0.1;
  int nk = 1;
  int max_epochs = atoi(argv[6]);

  if (argc>7) nk = atoi(argv[7]);
  if (argc>8) lr = atof(argv[8]);

  //generate random data
  int ***train_X;
  int ***test_X;
  if (ndat>0) {
    //generate random data
    train_X = new int**[ndat];
    for (int i=0; i<ndat; i++) {
      train_X[i] = new int*[nvis];
      for ( int j=0; j<nvis; j++) {
        train_X[i][j] = new int[nval];
        for ( int k=0; k<nval; k++) {
          train_X[i][j][k] = 0;
        }
        train_X[i][j][uniformint(0, nval)] = 1;
      }
    }
    ntest = ndat;
    test_X = new int**[ntest];
    for (int i=0; i<ntest; i++) {
      test_X[i] = new int*[nvis];
      for ( int j=0; j<nvis; j++) {
        test_X[i][j] = new int[nval];
        for ( int k=0; k<nval; k++) {
          test_X[i][j][k] = 0;
        }
        test_X[i][j][uniformint(0, nval)] = 1;
      }
    }
  }
  else if (ndat==0) {
    //load data from file
    ifstream fp;
    fp.open(argv[4]);
    if (!fp) {
      //not found
      cerr << "Data file not found!" << endl;
      return 1;
    }
    else {
      int nv = 0;
      string str;
      fp >> ndat >> nv >> ntest;
      cerr << "Loading data from " << argv[4] << " " << ndat << ", " << nv << endl;
      getline(fp, str);
      if (nvis == nv) {
        //load
        train_X = new int**[ndat];
        for (int i=0; i<ndat; i++) {
          train_X[i] = new int*[nvis];
          getline(fp, str);
          for (int j=0; j<nvis; j++) {
            train_X[i][j] = new int[nval];
            //cout << "db: " << str.c_str()[j];
            for (int l=0; l<nval; l++) {
              if (l==str.c_str()[j]-'0') {
                train_X[i][j][l] = 1;
              }
              else {
                train_X[i][j][l] = 0;
              }
            }
          }
        }
        //test
        test_X = new int**[ntest];
        for (int i=0; i<ntest; i++) {
          test_X[i] = new int*[nvis];
          string str;
          getline(fp, str);
          for (int j=0; j<nvis; j++) {
            test_X[i][j] = new int[nval];
            //cout << "db: " << str.c_str()[j];
            for (int l=0; l<nval; l++) {
              if (l==str.c_str()[j]-'0') {
                test_X[i][j][l] = 1;
              }
              else {
                test_X[i][j][l] = 0;
              }
            }
          }
        }
      }
      else {
        cerr << "Data shape doesn't match!" << endl;
        return 1;
      }
    }
    fp.close();
  }

  //init
  RBM rbm(nvis, nhid, nval, nk);

  //load weights if file exists
  bool weights_loaded = false;
  ifstream fp;
  fp.open(argv[5]);
  if (fp) {
    //overwrite random weights
    rbm.load(fp);
    weights_loaded = true;
  }
  fp.close();

  //go
  if (max_epochs>0) {
    rbm.train(train_X, ndat, test_X, ntest, max_epochs, lr);

    if (weights_loaded) {
      stringstream fn;
      fn << argv[5] << ".new";
      cout << "Save *NEW* weights in " << fn.str() << endl;
      ofstream fp;
      fp.open(fn.str().c_str());
      if (fp) rbm.save(fp);
      else {
        cerr << "Can't write new weights" << endl;
      }
    }
    else {
      cout << "Save weights in " << argv[5] << endl;
      ofstream fp;
      fp.open(argv[5]);
      if (fp) rbm.save(fp);
      else {
        cerr << "Can't write weights" << endl;
      }
    }
  }
  else {
    rbm.daydream(-max_epochs);
  }

  //cleanup
  for (int i=0; i<ndat; i++) {
    for (int j=0; j<nvis; j++) {
      delete [] train_X[i][j];
    }
    delete [] train_X[i];
  }
  delete [] train_X;
  for (int i=0; i<ntest; i++) {
    for (int j=0; j<nvis; j++) {
      delete [] test_X[i][j];
    }
    delete [] test_X[i];
  }
  delete [] test_X;
}

//todo: options; load data; load/save weights; CD-k
