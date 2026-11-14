#include <math.h>
#include <stdlib.h>
#include <stdint.h>
/* q_i = selected copies of cow-day i. Partial boundary clusters consume only
   remaining slots. Case detection is the max selected clone prefix over days. */
double evaluate(int64_t n, int64_t nd, int64_t nc, int64_t ncase,
 const int64_t *cow, const int64_t *day, const int64_t *order,
 const int64_t *starts, const double *wc, const int64_t *ci,
 const int64_t *casecow, double k, int mode,
 const double *total, double *selected_out){
 double *h=calloc(ncase>0?ncase:1,sizeof(double));double num=0,den=0;
 for(int64_t z=0;z<n;z++)if(ci[z]>=0 && mode==1)den+=wc[cow[z]];
 if(mode==2)for(int64_t j=0;j<ncase;j++)den+=wc[casecow[j]];
 for(int64_t d=0;d<nd;d++){
  int64_t a=starts[d],b=d+1<nd?starts[d+1]:n;
  double left=ceil(total[d]*k);if(left<1)left=1;
  for(int64_t j=a;j<b && left>0;j++){
   int64_t i=order[j];double q=wc[cow[i]];if(q>left)q=left;left-=q;
   if(selected_out)selected_out[i]=q;
   if(ci[i]>=0){if(mode==1)num+=q;else if(q>h[ci[i]])h[ci[i]]=q;}
  }
 }
 if(mode==2)for(int64_t j=0;j<ncase;j++)num+=h[j];
 free(h);return den>0?num/den:NAN;
}
