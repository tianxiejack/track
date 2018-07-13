#include "Kalman.h"
#include <math.h>
#include<stdio.h>
#include <stdlib.h>
#include<string.h>

CKalman::CKalman()
{
	state_pre = NULL;
	state_post = NULL;
	measure_param = NULL;
	transition_matrix = NULL;
	process_noise_cov = NULL;
	measurement_matrix = NULL;
	measurement_noise_cov = NULL;
	error_cov_pre = NULL;
	error_cov_post = NULL;
	gain = NULL;
	control_matrix = NULL;

	B_Uk = NULL;
	A_Pk = NULL;
	A_T = NULL;
	APA_T = NULL;

	H_T = NULL;
	Pk_Ht = NULL;
	Pk_Ht_R = NULL;
	Pk_Ht_R_Inv = NULL;
	H_Xk = NULL;
	Kk_H_Xk = NULL;
	H_Pk = NULL;
	Kk_H_Pk = NULL;
	deltat = 0.04; 

	memSpace = NULL;

	m_bInited = FALSE;
}

CKalman::~CKalman()
{
	KalmanClose();
	m_bInited = FALSE;
}

int CKalman::KalmanOpen(int D, int M, int C )
{
	int ptrPos = 0;
	if( D <= 0 || M <= 0 ){
		//AfxMessageBox("state and measurement vectors must have positive number of dimensions! ");	
		m_bInited = FALSE;
		return -1;
	}
	if( C < 0 ){
        //AfxMessageBox("No Control!");
		m_bInited = FALSE;
		return -2;
	}
    DP = D;
    MP = M;
    CP = C;

    if(memSpace == NULL){
    	memSpace = (double*)new double[512];
    }
    memset(memSpace, 0, sizeof(double)*512);
	if(state_pre == NULL){
		state_pre = memSpace+ptrPos;//new double[DP * 1];
		ptrPos +=DP;
//		memset( state_pre ,  0 , sizeof (*state_pre) );
	}
	if(state_post == NULL){
		state_post = memSpace+ptrPos;//new double[DP * 1];
		ptrPos +=DP;
//		memset( state_post , 0 , sizeof (*state_post) );
	}
	if (measure_param == NULL){
		measure_param = memSpace+ptrPos;//new double[MP * 1];
		ptrPos +=MP;
//		memset( measure_param , 0 , sizeof (*measure_param) );
	}
	if(transition_matrix == NULL){
		transition_matrix = memSpace+ptrPos;//new double[DP * DP];
		ptrPos +=DP*DP;
//		memset( transition_matrix , 0 , sizeof (*transition_matrix) );
	}
	if(process_noise_cov == NULL){
		process_noise_cov = memSpace+ptrPos;//new double[DP * DP];
		ptrPos +=DP*DP;
//		memset( process_noise_cov , 0 , sizeof (*process_noise_cov) );
	}
	if(measurement_matrix == NULL){
		measurement_matrix = memSpace+ptrPos;//new double[MP * DP];
		ptrPos +=MP*DP;
//		memset( measurement_matrix , 0 , sizeof (*measurement_matrix) );
	}
	if(measurement_noise_cov == NULL){
		measurement_noise_cov = memSpace+ptrPos;//new double[MP * MP];
		ptrPos +=MP*MP;
//		memset( measurement_noise_cov , 0 , sizeof (*measurement_noise_cov) );
	}
	if(error_cov_pre == NULL){
		error_cov_pre = memSpace+ptrPos;//new double[DP * DP];
		ptrPos +=DP*DP;
//		memset( error_cov_pre , 0 , sizeof (*error_cov_pre) );
	}
	if(error_cov_post == NULL){
		error_cov_post = memSpace+ptrPos;//new double[DP * DP];
		ptrPos +=DP*DP;
//		memset( error_cov_post , 0 , sizeof (*error_cov_post) );
	}
	if(gain == NULL){
		gain = memSpace+ptrPos;//new double[DP * MP];
		ptrPos +=DP*MP;
//		memset( gain , 0 , sizeof (*gain) );
	}
	if( CP > 0 )
    {
		if(control_matrix == NULL){
			control_matrix = memSpace+ptrPos;//new double[ DP * CP ];
			ptrPos +=DP*CP;
//			memset( control_matrix , 0 , sizeof (*control_matrix) );
		}
    }

	if(B_Uk == NULL){
		B_Uk  = memSpace+ptrPos;//new double[ DP * MP ];
		ptrPos +=DP*MP;
//		memset( B_Uk ,  0 , sizeof (*B_Uk) );
	}
	if(A_Pk == NULL){
		A_Pk  = memSpace+ptrPos;//new double[ DP * DP ];
		ptrPos +=DP*DP;
//		memset( A_Pk ,  0 , sizeof (*A_Pk) );
	}
	if(A_T == NULL){
		A_T   = memSpace+ptrPos;//new double[ DP * DP ];
		ptrPos +=DP*DP;
//		memset( A_T ,  0 , sizeof (*A_T) );
	}
	if(APA_T == NULL){
		APA_T = memSpace+ptrPos;//new double[ DP * DP ];
		ptrPos +=DP*DP;
//		memset( APA_T ,  0 , sizeof (*APA_T) );
	}
	
	if(H_T == NULL){
		H_T       = memSpace+ptrPos;//new double [ DP * MP ];
		ptrPos +=DP*MP;
//		memset( H_T ,  0 , sizeof (*H_T) );
	}
	if(Pk_Ht == NULL){
		Pk_Ht     = memSpace+ptrPos;//new double [ DP * MP ];
		ptrPos +=DP*MP;
//		memset( Pk_Ht ,  0 , sizeof (*Pk_Ht) );
	}
	if(Pk_Ht_R == NULL){
		Pk_Ht_R   = memSpace+ptrPos;//new double [ MP * MP ];
		ptrPos +=MP*MP;
//		memset( Pk_Ht_R ,  0 , sizeof (*Pk_Ht_R) );
	}
	if(Pk_Ht_R_Inv == NULL){
		Pk_Ht_R_Inv = memSpace+ptrPos;//new double [ MP * MP ];
		ptrPos +=MP*MP;
//		memset( Pk_Ht_R_Inv ,  0 , sizeof (*Pk_Ht_R_Inv) );
	}
	if(H_Xk == NULL){
		H_Xk      = memSpace+ptrPos;//new double [ MP *  1 ];
		ptrPos +=MP;
//		memset( H_Xk ,  0 , sizeof (*H_Xk) );
	}
	if(Kk_H_Xk == NULL){
		Kk_H_Xk   = memSpace+ptrPos;//new double [ DP *  1 ];
		ptrPos +=DP;
//		memset( Kk_H_Xk ,  0 , sizeof (*Kk_H_Xk) );
	}
	if(H_Pk == NULL){
		H_Pk      = memSpace+ptrPos;//new double [ MP * DP ];
		ptrPos +=MP*DP;
//		memset( H_Pk ,  0 , sizeof (*H_Pk) );
	}
	if(Kk_H_Pk == NULL){
		Kk_H_Pk   = memSpace+ptrPos;//new double [ DP * DP ];
		ptrPos +=DP*DP;
//		memset( Kk_H_Pk ,  0 , sizeof (*Kk_H_Pk) );
	}
	m_bInited = TRUE;
	return 0;
}

void CKalman::KalmanClose()
{
#if 1
	if(memSpace != NULL){
		delete []memSpace;
		memSpace = NULL;
	}
	state_pre = NULL;
	state_post = NULL;
	measure_param = NULL;
	transition_matrix = NULL;
	process_noise_cov = NULL;
	measurement_matrix = NULL;
	measurement_noise_cov = NULL;
	error_cov_pre = NULL;
	error_cov_post = NULL;
	gain = NULL;
	control_matrix = NULL;

	B_Uk = NULL;
	A_Pk = NULL;
	A_T = NULL;
	APA_T = NULL;

	H_T = NULL;
	Pk_Ht = NULL;
	Pk_Ht_R = NULL;
	Pk_Ht_R_Inv = NULL;
	H_Xk = NULL;
	Kk_H_Xk = NULL;
	H_Pk = NULL;
	Kk_H_Pk = NULL;
#else
	if(state_pre != NULL){
		delete [] state_pre ;        state_pre = NULL;
	}
	if(state_post != NULL){
		delete [] state_post ;       state_post = NULL;
	}
	if (measure_param != NULL){
		delete [] measure_param ;       measure_param = NULL;
	}
	if(transition_matrix != NULL){
		delete [] transition_matrix; transition_matrix = NULL;
	}
	if(CP >0)
	{
		if(control_matrix != NULL){
			delete [] control_matrix ; control_matrix = NULL;
		}
	}
	if(measurement_matrix != NULL){
		delete [] measurement_matrix;  measurement_matrix = NULL;
	}
	if(process_noise_cov != NULL){
		delete [] process_noise_cov;   process_noise_cov = NULL;
	}
	if(measurement_noise_cov != NULL){
		delete [] measurement_noise_cov; measurement_noise_cov = NULL;
	}
	if(error_cov_pre != NULL){
		delete [] error_cov_pre;       error_cov_pre = NULL;
	}
	if(gain != NULL){
		delete [] gain; gain = NULL;
	}
	if(error_cov_post != NULL){
		delete [] error_cov_post;      error_cov_post = NULL;
	}
	
	if(B_Uk != NULL){
		delete [] B_Uk;  B_Uk = NULL;
	}
	if(A_Pk != NULL){
		delete [] A_Pk;  A_Pk = NULL;
	}
	if(A_T != NULL){
		delete [] A_T;   A_T = NULL;
	}
	if(APA_T != NULL){
		delete [] APA_T; APA_T = NULL;
	}
	
	if(H_T != NULL){
		delete [] H_T;      H_T = NULL;
	}
	if(Pk_Ht != NULL){
		delete [] Pk_Ht;    Pk_Ht = NULL;
	}
	if(Pk_Ht_R != NULL){
		delete [] Pk_Ht_R;  Pk_Ht_R = NULL;
	}
	if(Pk_Ht_R_Inv != NULL){
		delete [] Pk_Ht_R_Inv;  Pk_Ht_R_Inv = NULL;
	}
	if(H_Xk != NULL){
		delete [] H_Xk;     H_Xk = NULL;
	}
	if(Kk_H_Xk != NULL){
		delete [] Kk_H_Xk;  Kk_H_Xk = NULL;
	}
	if(H_Pk != NULL){
		delete [] H_Pk;     H_Pk = NULL;
	}
	if(Kk_H_Pk != NULL){
		delete [] Kk_H_Pk;  Kk_H_Pk = NULL;
	}
#endif
}

#if 0
void CKalman::KalmanInitParam(int x_centre, int y_centre, double DeltaT)
{
	if (!m_bInited){
		return;
	}
	int x, y;
	
	for ( y = 0; y < DP; y++ ){
		for ( x = 0; x < DP; x++ ){
			process_noise_cov[y*DP+x] = 0.0;
		}
	}
	for ( y = 0; y < DP; y++ ){
		process_noise_cov[y*DP+y] = 1.0;  
	}
	
	for ( y = 0; y < MP; y++ ){
		for ( x = 0; x < MP; x++ ){
			measurement_noise_cov[y*MP+x] = 0.0;
		}
	}
	for ( y = 0; y < MP; y++ ){
		measurement_noise_cov[y*MP+y] = 1.0;  
	}
	
	for ( y = 0; y < DP; y++ ){
		for ( x = 0; x < DP; x++ ){
			error_cov_post[y*DP+x] = 0.0;
		}
	}
	for ( y = 0; y < DP; y++ ){
		error_cov_post[y*DP+y] = 10.0;  
	}
	
	for ( y = 0; y < DP; y++ ){
		for ( x = 0; x < DP; x++ ){
			transition_matrix[y*DP+x] = 0.0;
		}
	}
	for ( y = 0; y < DP; y++ ){
		transition_matrix[y*DP+y] = 1.0;  
	}
	transition_matrix[0*DP+2] = deltat;
	transition_matrix[1*DP+3] = deltat;

	for ( y = 0; y < MP; y++ ){
		for ( x = 0; x < DP; x++ ){
			measurement_matrix[y*DP+x] = 0.0; 
		}
	}
//	measurement_matrix[0*DP+0] = 1.0;
//	measurement_matrix[1*DP+1] = 1.0;
	for (y=0; y<MP; y++){
		measurement_matrix[y*DP+y] = 1.0;
	}

	
	measure_param[0] = (float)x_centre;
	measure_param[1] = (float)y_centre;
	
	state_post[0] = (float)x_centre;
	state_post[1] = (float)y_centre;
	state_post[2] = 0.0;
	state_post[3] = 0.0;

	deltat = DeltaT;
}

#else

void CKalman::KalmanInitParam(double x_centre, double y_centre, double DeltaT, double Q, double R)
{
	if (!m_bInited){
		return;
	}
	int x, y;
	
	for ( y = 0; y < DP; y++ ){
		for ( x = 0; x < DP; x++ ){
			process_noise_cov[y*DP+x] = 0.0;//Q
		}
	}

	process_noise_cov[0*DP+0] = Q;//0.00001;//4.0;
	process_noise_cov[1*DP+1] = Q;//0.00001;//4.0;
	process_noise_cov[2*DP+2] = Q;//0.00001;//9.0;
	process_noise_cov[3*DP+3] = Q;//0.00001;//9.0;


	for ( y = 0; y < MP; y++ ){
		for ( x = 0; x < MP; x++ ){
			measurement_noise_cov[y*MP+x] = 0.0;//R
		}
	}

	measurement_noise_cov[0*MP+0] = R;//0.25;//4.0;
	measurement_noise_cov[1*MP+1] = R;//0.25;//4.0;
	
	for ( y = 0; y < DP; y++ ){
		for ( x = 0; x < DP; x++ ){
			error_cov_post[y*DP+x] = 0.0;
		}
	}
	for ( y = 0; y < DP; y++ ){
		error_cov_post[y*DP+y] = 1.0;  
	}
	
	for ( y = 0; y < DP; y++ ){
		for ( x = 0; x < DP; x++ ){
			transition_matrix[y*DP+x] = 0.0;
		}
	}
	for ( y = 0; y < DP; y++ ){
		transition_matrix[y*DP+y] = 1.0; 
	}
	transition_matrix[0*DP+2] = DeltaT;
	transition_matrix[1*DP+3] = DeltaT;

	for ( y = 0; y < MP; y++ ){
		for ( x = 0; x < DP; x++ ){
			measurement_matrix[y*DP+x] = 0.0; 
		}
	}
	for (y=0; y<MP; y++){
		measurement_matrix[y*DP+y] = 1.0;
	}
	
	measure_param[0] = x_centre;
	measure_param[1] = y_centre;
	
	state_post[0] = x_centre;
	state_post[1] = y_centre;
	state_post[2] = 0.0;
	state_post[3] = 0.0;
	
	deltat = DeltaT;
}
#endif

void CKalman::KalmanPredict( double * control )//control-->u(k)
{
	if (!m_bInited){
		return;
	}
	/* update the state */
	/* x'(k) = A*x(k) */
	MatrixMultiply( transition_matrix, state_post, DP , DP , 1 , state_pre );

	if( control!=NULL && CP > 0 ){
		/* x'(k) = x'(k) + B*u(k) */
		MatrixMultiply( control_matrix, control, DP , CP , 1 , B_Uk);
		MatrixAddition( state_pre, B_Uk, DP, 1, state_pre);
	}
	
	/* update error covariance matrices */
	/* A_Pk = A*P(k) */
	MatrixMultiply( transition_matrix, error_cov_post, DP, DP, DP, A_Pk);
	
	/* P'(k) = A_Pk*At + Q */
	MatrixTranspose(transition_matrix, DP, DP, A_T);
	MatrixMultiply(A_Pk, A_T, DP, DP, DP, APA_T);
	MatrixAddition(APA_T, process_noise_cov, DP, DP, error_cov_pre);
}
	
void CKalman::KalmanCorrect( double * measurement )
{
	if (!m_bInited){
		return;
	}
	if( measurement == NULL)
	{
		printf("Measurement is Null!!! \n");
	}
	for (int i=0; i<MP; i++){
		measure_param[i] = measurement[i];
	}
	/* H_T = Ht*/
	MatrixTranspose( measurement_matrix , MP , DP , H_T);
	/* Pk_Ht = P'(k) * H_T */
    MatrixMultiply( error_cov_pre, H_T, DP , DP , MP , Pk_Ht);
	
    /* Pk_Ht_R = H*Pk_Ht + R */
    MatrixMultiply( measurement_matrix ,Pk_Ht , MP , DP , MP , Pk_Ht_R);
	MatrixAddition( Pk_Ht_R , measurement_noise_cov , MP , MP , Pk_Ht_R);
	
    /* Pk_Ht_R_Inv = inv(Pk_Ht_R) */
#if 0
    MatrixInversion( Pk_Ht_R , MP, Pk_Ht_R_Inv);   
#else
    MatrixCopy(Pk_Ht_R_Inv, Pk_Ht_R, MP, MP);
	MatrixBrinv(Pk_Ht_R_Inv, MP);
#endif
    
    /* K(k) = Pk_Ht * Pk_Ht_R_Inv  */
    MatrixMultiply( Pk_Ht , Pk_Ht_R_Inv, DP , MP ,MP , gain);

    //update state_post
    /* H_Xk = z(k) - H*x'(k) */
	MatrixMultiply( measurement_matrix , state_pre , MP , DP , 1, H_Xk);
	MatrixSubtraction( measure_param , H_Xk , MP , 1, H_Xk);    
    /* x(k) = x'(k) + K(k)*H_Xk */
	MatrixMultiply( gain , H_Xk, DP , MP, 1, Kk_H_Xk );
    MatrixAddition( state_pre , Kk_H_Xk , DP ,1 , state_post);
 
	//update error_cov_post
    /* P(k) = P'(k) - K(k)*H* P'(k) */
    MatrixMultiply( measurement_matrix , error_cov_pre , MP , DP , DP , H_Pk );
	MatrixMultiply( gain , H_Pk , DP , MP, DP , Kk_H_Pk );
	MatrixSubtraction( error_cov_pre , Kk_H_Pk , DP ,DP , error_cov_post);
}

void CKalman::MatrixMultiply(double* A, double* B, int m, int p, int n, double* C)
{
	// A = input matrix (m x p)
	// B = input matrix (p x n)
	// m = number of rows in A
	// p = number of columns in A = number of rows in B
	// n = number of columns in B
	// C = output matrix = A*B (m x n)
	int i, j, k;
	for (i=0;i<m;i++)
	{
		for(j=0;j<n;j++)
		{
			C[n*i+j]=0;
			for (k=0;k<p;k++)
			{
				C[n*i+j]= C[n*i+j]+A[p*i+k]*B[n*k+j];
			}
		}
	}
}

void CKalman::MatrixAddition(double* A, double* B, int m, int n, double* C)
{
	// A = input matrix (m x n)
	// B = input matrix (m x n)
	// m = number of rows in A = number of rows in B
	// n = number of columns in A = number of columns in B
	// C = output matrix = A+B (m x n)
	int i, j;
	for (i=0;i<m;i++)
	{
		for(j=0;j<n;j++)
		{
			C[n*i+j]=A[n*i+j]+B[n*i+j];
		}
	}
}

void CKalman::MatrixSubtraction(double* A, double* B, int m, int n, double* C)
{
	// A = input matrix (m x n)
	// B = input matrix (m x n)
	// m = number of rows in A = number of rows in B
	// n = number of columns in A = number of columns in B
	// C = output matrix = A-B (m x n)
	int i, j;
	for (i=0;i<m;i++)
	{
		for(j=0;j<n;j++)
		{
			C[n*i+j]=A[n*i+j]-B[n*i+j];
		}
	}
}

void CKalman::MatrixTranspose(double* A, int m, int n, double* C)
{
	// A = input matrix (m x n)
	// m = number of rows in A
	// n = number of columns in A
	// C = output matrix = the transpose of A (n x m)
	int i, j;
	for (i=0;i<m;i++)
	{
		for(j=0;j<n;j++)
		{
			C[m*j+i]=A[n*i+j];
		}
	}
}

int CKalman::MatrixInversion(double* A, int n, double* AInverse)
{
	// A = input matrix (n x n)
	// n = dimension of A 
	// AInverse = inverted matrix (n x n)
	// This function inverts a matrix based on the Gauss Jordan method.
	// The function returns 1 on success, 0 on failure.
	int i, j, iPass, imx, icol, irow;
	double det, temp, pivot, factor;
/*	double* ac = (double*)calloc(n*n, sizeof(double));*///modified by zhong
	det = 1;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			AInverse[n*i+j] = 0;
/*			ac[n*i+j] = A[n*i+j];*/
		}
		AInverse[n*i+i] = 1;
	}
	// The current pivot row is iPass.  
	// For each pass, first find the maximum element in the pivot column.
	for (iPass = 0; iPass < n; iPass++)
	{
		imx = iPass;
		for (irow = iPass; irow < n; irow++)
		{
			if (fabs(A[n*irow+iPass]) > fabs(A[n*imx+iPass]))
				imx = irow;
		}
		// Interchange the elements of row iPass and row imx in both A and AInverse.
		if (imx != iPass)
		{
			for (icol = 0; icol < n; icol++)
			{
				temp = AInverse[n*iPass+icol];
				AInverse[n*iPass+icol] = AInverse[n*imx+icol];
				AInverse[n*imx+icol] = temp;
				if (icol >= iPass)
				{
					temp = A[n*iPass+icol];
					A[n*iPass+icol] = A[n*imx+icol];
					A[n*imx+icol] = temp;
				}
			}
		}
		// The current pivot is now A[iPass][iPass].
		// The determinant is the product of the pivot elements.
		pivot = A[n*iPass+iPass];
		det = det * pivot;
		if (det == 0) 
		{
/*			free(ac);*/
			return 0;
		}
		for (icol = 0; icol < n; icol++)
		{	
			// Normalize the pivot row by dividing by the pivot element.
			AInverse[n*iPass+icol] = AInverse[n*iPass+icol] / pivot;
			if (icol >= iPass)
				A[n*iPass+icol] = A[n*iPass+icol] / pivot;
		}
		for (irow = 0; irow < n; irow++)
		{	
			// Add a multiple of the pivot row to each row.  The multiple factor 
			// is chosen so that the element of A on the pivot column is 0.
			if (irow != iPass)
				factor = A[n*irow+iPass];
			for (icol = 0; icol < n; icol++)
			{
				if (irow != iPass)
				{
					AInverse[n*irow+icol] -= factor * AInverse[n*iPass+icol];
					A[n*irow+icol] -= factor * A[n*iPass+icol];
				}
			}
		}
	}
/*	free(ac);*/
	return 1;
}

int CKalman::MatrixBrinv( double * A, int n)
{
	int * is, * js, i, j, k, l, u, v;
	double d,p;

	is = (int *)malloc( n*sizeof(int) );
	js = (int *)malloc( n*sizeof(int) );

	for ( k = 0; k < n; k++ )
	{ 
		d = 0.0;
		for ( i = k; i < n; i++ )
		{
			for ( j = k; j < n; j++ )
			{ 
				l = i*n+j;
				p = fabs(A[l]);
				if ( p > d )
				{ 
					d = p; is[k] = i; js[k] = j;
				}
			}
		}
		if ( d+1.0 == 1.0 ) 
		{ 
			free( is ); 
			free( js ); 
			return( 0 );
		}
		if ( is[k] != k )
		{
			for ( j = 0; j < n; j++ )
			{ 
				u = k*n+j;
				v = is[k]*n+j;
				p = A[u]; A[u] = A[v]; A[v] = p;
			}
		}
		if ( js[k] != k )
		{
			for ( i = 0; i < n; i++ )
			{ 
				u = i*n+k;
				v = i*n+js[k];
				p = A[u]; A[u] = A[v]; A[v] = p;
			}
		}
		l = k*n+k;
		A[l] = 1.0/A[l];
		for ( j = 0; j < n; j++ )
		{
			if ( j != k )
			{ 
				u = k*n+j;
				A[u] = A[u]*A[l];
			}
		}
		for ( i = 0; i < n; i++ )
		{
			if ( i != k )
			{
				for ( j = 0; j < n; j++ )
				{
					if ( j != k )
					{ 
						u = i*n+j;
						A[u] = A[u] - A[i*n+k]*A[k*n+j];
					}
				}
			}
		}
		for ( i = 0; i < n; i++ )
		{
			if ( i != k )
			{ 
				u = i*n+k;
				A[u] = -A[u]*A[l];
			}
		}
	}
	for ( k = n-1; k >= 0; k-- )
	{ 
		if ( js[k] != k )
		{
			for ( j = 0; j <= n-1; j++ )
			{ 
				u = k*n+j;
				v = js[k]*n+j;
				p = A[u]; A[u] = A[v]; A[v] = p;
			}
		}
		if ( is[k] != k )
		{
			for ( i = 0; i < n; i++ )
			{ 
				u = i*n+k;
				v = i*n+is[k];
				p = A[u]; A[u] = A[v]; A[v] = p;
			}
		}
	}
	free( is );
	free( js );

	return(1);
}

void CKalman::MatrixCopy(double *A, double *B, int m, int n)
{
	memcpy(A, B, sizeof(double)*m*n);
}
