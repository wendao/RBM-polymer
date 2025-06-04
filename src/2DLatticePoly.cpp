#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define NTOT 1000
#define NSAVE 10
#define LMAX 4
#define LATMAX 1000
#define MAX 3

// Global variables
int ix[NTOT], iy[NTOT];
int lat[LATMAX][LATMAX];
int iu[LMAX], ju[LMAX];
int lij[MAX][MAX];

// Random number generator state
static long idum2 = 123456789;
static long iv[32] = {0};
static long iy_rng = 0;

// Function prototypes
double ran2(long *idum);
double ran(long *iseed);
void setijl();
void ini(int nuse, long *iseed, int iread);
void sweep(int nuse, long *iseed, double *rate, int ixx);
void anls(int nuse, double *end2);
void outdat(int nuse, int irite);

// Random number generator (Numerical Recipes ran2)
double ran2(long *idum) {
    const long IM1 = 2147483563;
    const long IM2 = 2147483399;
    const double AM = 1.0 / IM1;
    const long IMM1 = IM1 - 1;
    const long IA1 = 40014;
    const long IA2 = 40692;
    const long IQ1 = 53668;
    const long IQ2 = 52774;
    const long IR1 = 12211;
    const long IR2 = 3791;
    const long NTAB = 32;
    const long NDIV = 1 + IMM1 / NTAB;
    const double EPS = 1.2e-7;
    const double RNMX = 1.0 - EPS;
    
    long j, k;
    
    if (*idum <= 0) {
        *idum = (-(*idum) > 1) ? -(*idum) : 1;
        idum2 = *idum;
        for (j = NTAB + 8; j >= 1; j--) {
            k = (*idum) / IQ1;
            *idum = IA1 * ((*idum) - k * IQ1) - k * IR1;
            if (*idum < 0) *idum += IM1;
            if (j <= NTAB) iv[j-1] = *idum;
        }
        iy_rng = iv[0];
    }
    
    k = (*idum) / IQ1;
    *idum = IA1 * ((*idum) - k * IQ1) - k * IR1;
    if (*idum < 0) *idum += IM1;
    
    k = idum2 / IQ2;
    idum2 = IA2 * (idum2 - k * IQ2) - k * IR2;
    if (idum2 < 0) idum2 += IM2;
    
    j = 1 + iy_rng / NDIV;
    iy_rng = iv[j-1] - idum2;
    iv[j-1] = *idum;
    if (iy_rng < 1) iy_rng += IMM1;
    
    double result = AM * iy_rng;
    return (result > RNMX) ? RNMX : result;
}

double ran(long *iseed) {
    return ran2(iseed);
}

// Setup direction vectors
void setijl() {
    // Direction vectors (up, left, down, right)
    iu[0] = 0;  ju[0] = -1;  // up
    iu[1] = -1; ju[1] = 0;   // left
    iu[2] = 0;  ju[2] = 1;   // down
    iu[3] = 1;  ju[3] = 0;   // right
    
    // Initialize lookup table
    for (int i = 0; i < MAX; i++) {
        for (int j = 0; j < MAX; j++) {
            lij[i][j] = 0;
        }
    }
    
    lij[1][0] = 1;  // (-1,0) -> direction 2
    lij[2][1] = 4;  // (0,1) -> direction 4
    lij[0][1] = 2;  // (0,-1) -> direction 1
    lij[1][2] = 3;  // (1,0) -> direction 3
}

// Initialize polymer chain
void ini(int nuse, long *iseed, int iread) {
    // Initialize chain at center
    ix[0] = LATMAX / 2;
    iy[0] = LATMAX / 2;
    
    // Generate zigzag chain
    for (int n = 1; n < nuse; n += 2) {
        if (n < nuse) {
            ix[n] = ix[n-1];
            iy[n] = iy[n-1] + 1;
        }
        if (n+1 < nuse) {
            ix[n+1] = ix[n] + 1;
            iy[n+1] = iy[n];
        }
    }
    
    // Read from file if requested
    if (iread == 0) {
        FILE *fp = fopen("Config.dat", "r");
        if (fp != NULL) {
            for (int n = 0; n < nuse; n++) {
                fscanf(fp, "%d %d", &ix[n], &iy[n]);
            }
            fclose(fp);
        }
    }
    
    // Initialize lattice
    for (int i = 0; i < LATMAX; i++) {
        for (int j = 0; j < LATMAX; j++) {
            lat[i][j] = 0;
        }
    }
    
    // Mark occupied sites
    for (int n = 0; n < nuse; n++) {
        lat[ix[n]][iy[n]] = 1;
    }
}

// Monte Carlo sweep
void sweep(int nuse, long *iseed, double *rate, int ixx) {
    *rate = 0.0;
    
    for (int icont = 0; icont < nuse; icont++) {
        // Select monomer
        int n = (int)(ran(iseed) * nuse);
        if (n >= nuse) n = nuse - 1;
        
        int ixnew, iynew;
        
        // Make a move
        if (n == 0) {  // First monomer
            int nran = (int)(ran(iseed) * LMAX);
            if (nran >= LMAX) nran = LMAX - 1;
            ixnew = ix[n+1] + iu[nran];
            iynew = iy[n+1] + ju[nran];
        }
        else if (n == nuse - 1) {  // Last monomer
            int nran = (int)(ran(iseed) * LMAX);
            if (nran >= LMAX) nran = LMAX - 1;
            ixnew = ix[n-1] + iu[nran];
            iynew = iy[n-1] + ju[nran];
        }
        else {  // Middle monomers
            // Check if straight (no move possible)
            if (ix[n-1] == ix[n+1] || iy[n-1] == iy[n+1]) {
                continue;
            }
            // Corner move: flip the bond
            ixnew = ix[n-1] + ix[n+1] - ix[n];
            iynew = iy[n-1] + iy[n+1] - iy[n];
        }
        
        // Check bounds
        if (ixnew < 0 || ixnew >= LATMAX || iynew < 0 || iynew >= LATMAX) {
            continue;
        }
        
        // Excluded volume check
        if (ixx != 0 && lat[ixnew][iynew] == 1) {
            continue;  // Reject move
        }
        
        // Accept move
        lat[ix[n]][iy[n]] = 0;
        ix[n] = ixnew;
        iy[n] = iynew;
        lat[ix[n]][iy[n]] = 1;
        *rate += 1.0;
    }
    
    *rate /= (double)nuse;
}

// Analysis routine
void anls(int nuse, double *end2) {
    // Calculate end-to-end distance squared
    *end2 = (ix[nuse-1] - ix[0]) * (ix[nuse-1] - ix[0]) + 
            (iy[nuse-1] - iy[0]) * (iy[nuse-1] - iy[0]);
    
    // Center the chain
    int ix0 = ix[nuse/2];
    int iy0 = iy[nuse/2];
    
    // Clear lattice
    for (int n = 0; n < nuse; n++) {
        lat[ix[n]][iy[n]] = 0;
    }
    
    // Move to center
    for (int n = 0; n < nuse; n++) {
        ix[n] = ix[n] - ix0 + LATMAX/2;
        iy[n] = iy[n] - iy0 + LATMAX/2;
    }
    
    // Re-mark lattice
    for (int n = 0; n < nuse; n++) {
        lat[ix[n]][iy[n]] = 1;
    }
}

// Output coordinates
void outdat(int nuse, int irite) {
    FILE *fp = fopen("Coordinates.dat", "a");
    if (fp == NULL) {
        printf("Error opening Coordinates.dat\n");
        return;
    }
    
    for (int n = 0; n < nuse; n++) {
        fprintf(fp, "%8d %8d %8d\n", ix[n], iy[n], n+1);
    }
    
    if (irite != 0) {
        fprintf(fp, "\n");
    }
    
    fclose(fp);
}

// Main program
int main() {
    char aname[NSAVE][8];
    double aa[NSAVE];
    
    // Initialize names
    strcpy(aname[0], "END2");
    strcpy(aname[1], "RATE");
    for (int i = 2; i < NSAVE; i++) {
        strcpy(aname[i], " ");
    }
    
    // Read parameters
    FILE *fp = fopen("Ini.dat", "r");
    if (fp == NULL) {
        printf("Error: Cannot open Ini.dat\n");
        printf("Creating default Ini.dat file...\n");
        
        // Create default parameter file
        fp = fopen("Ini.dat", "w");
        fprintf(fp, " 8                      ! N=POLYMERIZATION INDEX\n");
        fprintf(fp, " 111                    ! ISEED=RANDOM SEED\n");
        fprintf(fp, " 1000000,10000          ! MAX MC STEP AND MEASUREMENT GAP\n");
        fprintf(fp, " 1                      ! TO READ (0) OR TO GENERATE (1) Initial Config\n");
        fprintf(fp, " 1                      ! TO WRITE PROFILES (1) OR ONCE (0)\n");
        fprintf(fp, " 1                      ! SELF AVOIDING (1) OR NOT (0)\n");
        fclose(fp);
        
        fp = fopen("Ini.dat", "r");
    }
    
    int nuse, mcsmax, nd, iread, irite, ixx;
    long iseed;
    char line[256];
    
    // Read NUSE
    fgets(line, sizeof(line), fp);
    sscanf(line, "%d", &nuse);
    
    // Read ISEED
    fgets(line, sizeof(line), fp);
    sscanf(line, "%ld", &iseed);
    
    // Read MCSMAX and ND (comma separated)
    fgets(line, sizeof(line), fp);
    sscanf(line, "%d,%d", &mcsmax, &nd);
    
    // Read IREAD
    fgets(line, sizeof(line), fp);
    sscanf(line, "%d", &iread);
    
    // Read IRITE
    fgets(line, sizeof(line), fp);
    sscanf(line, "%d", &irite);
    
    // Read IXX
    fgets(line, sizeof(line), fp);
    sscanf(line, "%d", &ixx);
    
    fclose(fp);
    
    // Initialize
    ini(nuse, &iseed, iread);
    setijl();
    
    printf("**Starting MC simulations...MCSMAX=%d\n", mcsmax);
    printf("BLOCK");
    for (int kp = 0; kp < 10; kp++) {
        printf("%11s", aname[kp]);
    }
    printf("\n");
    
    // Main simulation loop
    for (int nblock = 1; nblock <= 100; nblock++) {
        int kcont = 0;
        int ncont = 0;
        
        for (int kkk = 0; kkk < NSAVE; kkk++) {
            aa[kkk] = 0.0;
        }
        
        for (int mcs = 1; mcs <= mcsmax/100; mcs++) {
            double rate;
            sweep(nuse, &iseed, &rate, ixx);
            
            ncont++;
            if (ncont == nd) {
                double end2;
                anls(nuse, &end2);
                
                if (nblock <= 10) {
                    continue;  // Skip to next block
                }
                
                outdat(nuse, irite);
                ncont = 0;
                kcont++;
                double fk = (double)kcont;
                
                aa[0] = aa[0] + (end2/(double)(nuse-1) - aa[0])/fk;
                aa[1] = aa[1] + (rate - aa[1])/fk;
            }
        }
        
        // Output results
        printf("%3d ", nblock);
        for (int k0 = 0; k0 < 10; k0++) {
            printf(" %10.4e", aa[k0]);
        }
        printf("\n");
    }
    
    return 0;
}
