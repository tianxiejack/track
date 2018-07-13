

#ifndef _OSA_RNG_H_
#define _OSA_RNG_H_

#include <osa.h>

typedef struct {

  Uint32 curRd;
  Uint32 curWr;
  Uint32 len;
  Uint32 count;

  Uint8 *queue;

  pthread_mutex_t lock;
  pthread_cond_t  condRd;
  pthread_cond_t  condWr;
  
} OSA_RngHndl;

int OSA_rngCreate(OSA_RngHndl *hndl, Uint32 maxLen);
int OSA_rngDelete(OSA_RngHndl *hndl);
int OSA_rngPut(OSA_RngHndl *hndl, Uint8 *data, Int32  nbytes, Uint32 timeout);
int OSA_rngGet(OSA_RngHndl *hndl, Uint8 *data, Int32  nMaxBytes, Uint32 timeout);
Uint32 OSA_rngGetQueuedCount(OSA_RngHndl *hndl);
Bool OSA_rngIsEmpty(OSA_RngHndl *hndl);

#endif /* _OSA_RNG_H_ */



