class RBM {
public:
  int N_vis;
  int N_val;
  int N_hid;
  bool debug_print;

  float ***W; // n_vis*n_h*n_val
  float **vb; // n_vis*n_val
  float *hb; // n_hid*n_val

  RBM(int, int, int, int);
  virtual ~RBM();
  void train(int ***, int, int ***, int, int, float); //ndat*n_vis
  void load(ifstream &);
  void save(ofstream &);
  void daydream(int);

  void check_status(int ***, int, int ***, int);
};
