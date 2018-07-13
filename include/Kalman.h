#ifndef  _KALMAN_H_
#define  _KALMAN_H_

typedef bool BOOL;
#define FALSE 0
#define TRUE 1

class CKalman
{
public:
    CKalman(); 
	~CKalman();
	int  KalmanOpen(int D, int M, int C );
	void KalmanClose();
	void KalmanInitParam(double x_centre, double y_centre, double DeltaT, double Q, double R);
	void KalmanPredict( double * control );
	void KalmanCorrect( double * measurement );

	void MatrixMultiply(double* A, double* B, int m, int p, int n, double* C);
	void MatrixAddition(double* A, double* B, int m, int n, double* C);
	void MatrixSubtraction(double* A, double* B, int m, int n, double* C);
	void MatrixTranspose(double* A, int m, int n, double* C);
	int  MatrixInversion(double* A, int n, double* AInverse);
	int  MatrixBrinv( double * A, int n );
	void MatrixCopy(double *A, double *B, int m, int n);

public:	
	int MP;  //number of measure vector dimensions
	int DP;  //number of state   vector dimensions
	int CP;  //number of control vector dimensions
	double *memSpace;

	double* state_pre;            //[DP * 1] 
	double* state_post;           //[DP * 1] 
	double* measure_param;        //[MP * 1]
	double* transition_matrix;    //[DP * DP] 
	double* control_matrix;       //[DP * CP] if (CP > 0) 
	double* measurement_matrix;   //[MP * DP]
	double* process_noise_cov;    //[DP * DP]
	double* measurement_noise_cov;//[MP * MP]
	double* error_cov_pre;        //[DP * DP]
	double* error_cov_post;       //[DP * DP]
	double* gain;                 //[DP * MP]

	double   deltat; // 
	BOOL     m_bInited;

public: //temporary  variable 
	double *B_Uk;		 //[ DP * 1  ]
	double *A_Pk;		 //[ DP * DP ]
	double *A_T;		 //[ DP * DP ]
	double *APA_T;		 //[ DP * DP ]

	double *H_T;         //[ DP * MP ]
	double *Pk_Ht;       //[ DP * MP ]
	double *Pk_Ht_R;     //[ MP * MP ]
	double *Pk_Ht_R_Inv; //[ MP * MP ]
	double *H_Xk;        //[ MP * 1  ]
	double *Kk_H_Xk;     //[ DP * 1  ]
	double *H_Pk;        //[ MP * DP ]
	double *Kk_H_Pk;     //[ DP * DP ]

};

#endif
