#include "gelu_lookup_kernel.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
namespace {
float reference(float x) {
    float u=x*x;u=u*x;u=u*0.044715f;u=u+x;u=u*0.79788458f;u=std::max(-5.0f,std::min(5.0f,u));
    const float z=u*u;
    float a=z+378;a=a*z;a=a+17325;a=a*z;a=a+135135;a=a*u;
    float b=z*28;b=b+3150;b=b*z;b=b+62370;b=b*z;b=b+135135;
    float t=std::max(-1.0f,std::min(1.0f,a/b));t=t+1;t=t*x;t=t*0.5f;return t;
}
void check(size_t count,bool inplace,int mode) {
    std::vector<float> source(count),expected(count),output(count),points(4097),table(8192);
    std::mt19937 random{unsigned(count)};
    for(size_t i=0;i<count;++i){source[i]=float(int(random()%2400001)-1200000)/100000.0f;expected[i]=reference(source[i]);}
    if(count>4){source[0]=-100;source[1]=-8;source[2]=0;source[3]=8;source[4]=100;for(size_t i=0;i<5;++i)expected[i]=reference(source[i]);}
    for(size_t i=0;i<points.size();++i)points[i]=reference(float(i)/256-8);
    tinyvit_gelu::coefficients(table.data(),points.data());
    if(inplace)output=source;
    float* destination=output.data();const float* input=inplace?output.data():source.data();
    if(mode==0)tinyvit_gelu::reciprocal(destination,input,count);
    else if(mode==1)tinyvit_gelu::apply(destination,input,count,table.data());
    else tinyvit_gelu::polynomial(destination,input,count);
    double maximum=0;
    for(size_t i=0;i<count;++i){const double error=std::abs(double(output[i])-expected[i]);maximum=std::max(maximum,error);
        if(!std::isfinite(output[i]) || error>5e-5+5e-5*std::abs(expected[i]))throw std::runtime_error("GELU scalar oracle mismatch");}
    std::cout<<"PASS count="<<count<<" inplace="<<inplace<<" mode="<<mode<<" maximum="<<maximum<<'\n';
}
}
int main(){try{for(int mode:{0,1,2})for(bool inplace:{false,true})for(size_t n:{size_t(1),size_t(3),size_t(4),size_t(7),size_t(8),size_t(17),size_t(4097),size_t(100003)})check(n,inplace,mode);
std::cout<<"PASS 48 GELU oracle cases\n";}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
