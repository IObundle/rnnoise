/* Copyright (c) 2008-2011 Octasic Inc.
                 2012-2017 Jean-Marc Valin */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <math.h>
#include "opus_types.h"
#include "common.h"
#include "arch.h"
#include "tansig_table.h"
#include "rnn.h"
#include "rnn_data.h"
#include <stdio.h>
#include "iob_rnn_acc_swreg.h"
#include "iob-rnn-acc.h"

int invert ( int x ){
    
    int res = ~(x) & (int)0x0ffffffff;
    return res;
    
    };


static OPUS_INLINE float tansig_approx(float x)
{
    int i;
    float y, dy;
    float sign=1;
    /* Tests are reversed to catch NaNs */
    if (!(x<8))
        return 1;
    if (!(x>-8))
        return -1;
#ifndef FIXED_POINT
    /* Another check in case of -ffast-math */
    if (celt_isnan(x))
       return 0;
#endif
    if (x<0)
    {
       x=-x;
       sign=-1;
    }
    i = (int)floor(.5f+25*x);
    x -= .04f*i;
    y = tansig_table[i];
    dy = 1-y*y;
    y = y + x*dy*(1 - y*x);
    return sign*y;
}

static OPUS_INLINE float sigmoid_approx(float x)
{
   return .5 + .5*tansig_approx(.5*x);
}

static OPUS_INLINE float relu(float x)
{
   return x < 0 ? 0 : x;
}

void compute_dense(const DenseLayer *layer, float *output, const float *input)
{
   int i, j;
   int N, M;
   int stride;
   M = layer->nb_inputs;
   N = layer->nb_neurons;
   stride = N;
   for (i=0;i<N;i++)
   {
      /* Compute update gate. */
      float sum = layer->bias[i];
      for (j=0;j<M;j++)
         sum += layer->input_weights[j*stride + i]*input[j];
      output[i] = WEIGHTS_SCALE*sum;
   }
   if (layer->activation == ACTIVATION_SIGMOID) {
      for (i=0;i<N;i++)
         output[i] = sigmoid_approx(output[i]);
   } else if (layer->activation == ACTIVATION_TANH) {
      for (i=0;i<N;i++)
         output[i] = tansig_approx(output[i]);
   } else if (layer->activation == ACTIVATION_RELU) {
      for (i=0;i<N;i++)
         output[i] = relu(output[i]);
   } else {
     *(int*)0=0;
   }
}

void compute_gru(const GRULayer *gru, float *state, const float *input)
{
   int i, j;
   int N, M;
   int stride;
   float z[MAX_NEURONS];
   float r[MAX_NEURONS];
   float h[MAX_NEURONS];
   M = gru->nb_inputs;
   N = gru->nb_neurons;
   stride = 3*N;

   /* IOB_RNN_ACC_SET_ACC_TESTW(int_a); */
  
  /* unsigned xxx = IOB_RNN_ACC_GET_ACC_TESTR(); */
  
  /* printf(" the XXX is %08x\n the converted %08x\n",int_a,xxx); */
  
  /* int thepowplus_2 = 65536; */
  
  /* int int_a_b; */
  /* if (xxx & 0x80000000) { */
  /*   int_a_b = xxx -1; */
  /*   int_a_b = invert(int_a_b); */
  /* }else */
  /*   int_a_b = xxx; */
  /* double int_a_f = (double) int_a_b / thepowplus_2; */
  
  /* printf(" the XXX after conversion is %f\n",int_a_f); */
  /* ******************************** */

   for (i=0;i<N;i++)
     {
       /* Compute update gate. */
       
       float sum = gru->bias[i];
       /* convert the values to in tto send to the verilog files */
       int int_a;
       double thepow = 5.960464477539063e-08;
       double thepowplus = 16777216;
       
       double bbb = sum;
       if (fabs(bbb) < thepow){
	 bbb = 0.0;
       }
       if (bbb <0)
	 {
	   bbb = bbb * -1;
	   int_a = invert((int)(round(bbb * thepowplus))) + 1;
	 }else
	 {
	   int_a = (int)(round(bbb * thepowplus));
	 }
       
       IOB_RNN_ACC_SET_ACC_TESTE(int_a);
       
       /* setq1 setwsum seten1 */
       /*   seten0 */
       for (j=0;j<M;j++)
	 {
	   /* sum += gru->input_weights[j*stride + i]*input[j]; */
	   int int_a;
	   double thepow = 5.960464477539063e-08;
	   double thepowplus = 16777216;
	   
	   double bbb = input[j];
	   if (fabs(bbb) < thepow){
	     bbb = 0.0;
	   }
	   if (bbb <0)
	     {
	       bbb = bbb * -1;
	       int_a = invert((int)(round(bbb * thepowplus))) + 1;
	     }else
	     {
	       int_a = (int)(round(bbb * thepowplus));
	     }
	   int enen=1;
	   IOB_RNN_ACC_SET_ACC_TESTEN(enen);

	   IOB_RNN_ACC_SET_ACC_TESTW(gru->input_weights[j*stride + i]);

	   IOB_RNN_ACC_SET_ACC_TESTE(int_a);

	   enen=0;
	   IOB_RNN_ACC_SET_ACC_TESTEN(enen);
	   
	   /* setq setw */
	   /*   seten */
	   /* there shoudl be conversion to int here */
	     
	 }
       
       for (j=0;j<N;j++){
         /* sum += gru->recurrent_weights[j*stride + i]*state[j]; */
	 int int_a;
	 double thepow = 5.960464477539063e-08;
	 double thepowplus = 16777216;
	 
	 double bbb = state[j];
	 if (fabs(bbb) < thepow){
	   bbb = 0.0;
	 }
	 if (bbb <0)
	   {
	     bbb = bbb * -1;
	       int_a = invert((int)(round(bbb * thepowplus))) + 1;
	   }else
	   {
	     int_a = (int)(round(bbb * thepowplus));
	   }
	 int enen=1;
	 IOB_RNN_ACC_SET_ACC_TESTEN(enen);
	 
	 IOB_RNN_ACC_SET_ACC_TESTW(gru->recurrent_weights[j*stride + i]);
	 
	 IOB_RNN_ACC_SET_ACC_TESTE(int_a);
	 
	 enen=0;
	 IOB_RNN_ACC_SET_ACC_TESTEN(enen);
	 
	 /* setq setw */
	 /*   seten */
	 /* there shoudl be conversion to int here */
	 
	 
	 
       }
       
       int enen=1;
       IOB_RNN_ACC_SET_ACC_TESTENREQ(enen);
	   
       unsigned xxx = IOB_RNN_ACC_GET_ACC_TESTR();
       
       enen=0;
       IOB_RNN_ACC_SET_ACC_TESTENREQ(enen);
       
       int thepowplus_2 = 65536;
       
       double int_a_b;
       if (xxx & 0x80000000) {
	 int_a_b = xxx -1;
	 int_a_b = invert(int_a_b);
       }else
	 int_a_b = xxx;
       double int_a_f = (double) int_a_b / thepowplus_2;
       
       sum = (float)int_a_f;
       /* printf(" the XXX after conversion is %f\n",int_a_f); */
       /* ******************************** */
       
       /* getter should bve here for sum */
       z[i] = sigmoid_approx(WEIGHTS_SCALE*sum);
     }
   for (i=0;i<N;i++)
     {
       /* Compute reset gate. */
       float sum = gru->bias[N + i];
       for (j=0;j<M;j++)
         sum += gru->input_weights[N + j*stride + i]*input[j];
       for (j=0;j<N;j++)
         sum += gru->recurrent_weights[N + j*stride + i]*state[j];
       r[i] = sigmoid_approx(WEIGHTS_SCALE*sum);
   }
   

   
   for (i=0;i<N;i++)
     {
       /* Compute output. */
       float sum = gru->bias[2*N + i];
       for (j=0;j<M;j++)
         sum += gru->input_weights[2*N + j*stride + i]*input[j];
       for (j=0;j<N;j++)
         sum += gru->recurrent_weights[2*N + j*stride + i]*state[j]*r[j];
       if (gru->activation == ACTIVATION_SIGMOID) sum = sigmoid_approx(WEIGHTS_SCALE*sum);
       else if (gru->activation == ACTIVATION_TANH) sum = tansig_approx(WEIGHTS_SCALE*sum);
       else if (gru->activation == ACTIVATION_RELU) sum = relu(WEIGHTS_SCALE*sum);
      else *(int*)0=0;
       h[i] = z[i]*state[i] + (1-z[i])*sum;
     }
   for (i=0;i<N;i++)
     state[i] = h[i];
}

#define INPUT_SIZE 42

void compute_rnn(RNNState *rnn, float *gains, float *vad, const float *input) {
  int i;
  float dense_out[MAX_NEURONS];
  float noise_input[MAX_NEURONS*3];
  float denoise_input[MAX_NEURONS*3];
  compute_dense(rnn->model->input_dense, dense_out, input);
  compute_gru(rnn->model->vad_gru, rnn->vad_gru_state, dense_out);
  compute_dense(rnn->model->vad_output, vad, rnn->vad_gru_state);
  for (i=0;i<rnn->model->input_dense_size;i++) noise_input[i] = dense_out[i];
  for (i=0;i<rnn->model->vad_gru_size;i++) noise_input[i+rnn->model->input_dense_size] = rnn->vad_gru_state[i];
  for (i=0;i<INPUT_SIZE;i++) noise_input[i+rnn->model->input_dense_size+rnn->model->vad_gru_size] = input[i];
  compute_gru(rnn->model->noise_gru, rnn->noise_gru_state, noise_input);

  for (i=0;i<rnn->model->vad_gru_size;i++) denoise_input[i] = rnn->vad_gru_state[i];
  for (i=0;i<rnn->model->noise_gru_size;i++) denoise_input[i+rnn->model->vad_gru_size] = rnn->noise_gru_state[i];
  for (i=0;i<INPUT_SIZE;i++) denoise_input[i+rnn->model->vad_gru_size+rnn->model->noise_gru_size] = input[i];
  compute_gru(rnn->model->denoise_gru, rnn->denoise_gru_state, denoise_input);
  compute_dense(rnn->model->denoise_output, gains, rnn->denoise_gru_state);
}
