

#include <osa_rng.h>

int OSA_rngCreate(OSA_RngHndl *hndl, Uint32 maxLen)
{
  pthread_mutexattr_t mutex_attr;
  pthread_condattr_t cond_attr;
  int status=OSA_SOK;

  hndl->curRd = hndl->curWr = 0;
  hndl->count = 0;
  hndl->len   = maxLen;
  hndl->queue = (Uint8 *)OSA_memAlloc(sizeof(Uint8)*hndl->len);
  
  if(hndl->queue==NULL) {
    OSA_ERROR("OSA_rngCreate() = %d \r\n", status);
    return OSA_EFAIL;
  }
 
  status |= pthread_mutexattr_init(&mutex_attr);
  status |= pthread_condattr_init(&cond_attr);  
  
  status |= pthread_mutex_init(&hndl->lock, &mutex_attr);
  status |= pthread_cond_init(&hndl->condRd, &cond_attr);    
  status |= pthread_cond_init(&hndl->condWr, &cond_attr);  

  if(status!=OSA_SOK)
    OSA_ERROR("OSA_rngCreate() = %d \r\n", status);
    
  pthread_condattr_destroy(&cond_attr);
  pthread_mutexattr_destroy(&mutex_attr);
    
  return status;
}

int OSA_rngDelete(OSA_RngHndl *hndl)
{
  if(hndl->queue!=NULL)
    OSA_memFree(hndl->queue);
    
  pthread_cond_destroy(&hndl->condRd);
  pthread_cond_destroy(&hndl->condWr);
  pthread_mutex_destroy(&hndl->lock);  
  
  return OSA_SOK;
}


int OSA_rngPut(OSA_RngHndl *hndl, Uint8 *data, Int32  nbytes, Uint32 timeout)
{
  int status = OSA_EFAIL;

  pthread_mutex_lock(&hndl->lock);

  while(1) {
    if( hndl->count + nbytes < hndl->len ) {
      int i;
      for(i=0; i<nbytes; i++){
	      hndl->queue[hndl->curWr] = data[i];
	      hndl->curWr = (hndl->curWr+1)%hndl->len;
	      hndl->count++;
      }
      status = OSA_SOK;
      pthread_cond_signal(&hndl->condRd);
      break;
    } else {
      if(timeout == OSA_TIMEOUT_NONE)
        break;

      status = pthread_cond_wait(&hndl->condWr, &hndl->lock);
    }
  }

  pthread_mutex_unlock(&hndl->lock);

  if(status == OSA_SOK)
  	return nbytes;
  
  return status;
}

int OSA_rngGet(OSA_RngHndl *hndl, Uint8 *data, Int32  nMaxBytes, Uint32 timeout)
{
  int status = OSA_EFAIL;
  int count = 0;
  
  pthread_mutex_lock(&hndl->lock);
  
  while(1) {
    if(hndl->count > 0 ) {
	  int i;

	  count  = (hndl->count < nMaxBytes ) ? hndl->count : nMaxBytes;

	  for(i=0; i<count; i++)
	  {
		if(data!=NULL) {
		data[i] = hndl->queue[hndl->curRd];
		}

		hndl->curRd = (hndl->curRd+1)%hndl->len;
		hndl->count--;
	  }

      status = OSA_SOK;
      pthread_cond_signal(&hndl->condWr);
      break;
    } else {
      if(timeout == OSA_TIMEOUT_NONE)
        break;

      status = pthread_cond_wait(&hndl->condRd, &hndl->lock);
    }
  }

  pthread_mutex_unlock(&hndl->lock);

  if(status == OSA_SOK)
  	return count;

  return status;
}


Uint32 OSA_rngGetQueuedCount(OSA_RngHndl *hndl)
{
  Uint32 queuedCount = 0;

  pthread_mutex_lock(&hndl->lock);
  queuedCount = hndl->count;
  pthread_mutex_unlock(&hndl->lock);
  return queuedCount;
}

Bool OSA_rngIsEmpty(OSA_RngHndl *hndl)
{
  Bool isEmpty;

  pthread_mutex_lock(&hndl->lock);
  if (hndl->count == 0)
  {
      isEmpty = TRUE;
  }
  else
  {
      isEmpty = FALSE;
  }
  pthread_mutex_unlock(&hndl->lock);

  return isEmpty;
}



