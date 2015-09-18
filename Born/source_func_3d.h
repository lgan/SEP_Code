#ifndef WAVELET_RTM_3D_H
#define WAVELET_RTM_3D_H 1
#include<sregf.h>
#include<hypercube_float.h>
#include "wavefield_insert_3d.h"
class source_func: public wavefield_insert_3d{
  public:
    source_func(){};
    source_func(char *tag);
    


    void set_source_file(char *tg){
      tag=tg;
    }
    virtual int get_points(bool e);
    ~source_func(){ delete wavelet;}
    void set_sources_3d(float sz, axis a_sx, axis as_y);
 /*   void set_sources(std::vector<float> *s_z, std::vector<float> *s_x, std::vector<float> *s_y){
      for(int i=0; i < (int)s_z->size();i++){
        sz.push_back(s_z->at(i)); sx.push_back(s_x->at(i)); sy.push_back(s_y->at(i));
      }
    }*/
//    hypercube_float *create_domain(int ishotx, int ishoty);
    hypercube_float *create_domain(int ishot);

    void set_compute_size(hypercube *dom, float aper,int nbt,int nb,int nby, int fat, int blocksize);
    int y_points(){ return ay.n;}
    int x_points(){ return ax.n;}
    int z_points(){ return az.n;}
    void get_source_func(hypercube_float *domain, int ishot,int nts, int *ilocs, float *vals);
    float get_dt(){return dt;}
    float dt;
    std::vector<float> sx,sz,sy;
    int nx,nz,ny,jt,jts,jtd;
    float aper;
    axis az,ax,ay;
    hypercube_float *wavelet;
    int nboundt, nbound, nbound_y;
    std::string tag;
    

};
#endif
