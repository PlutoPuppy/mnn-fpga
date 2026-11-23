#include "fused_attention_kernel.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void softmax(float* output,const float* input,size_t size) {
    const float maximum=*std::max_element(input,input+size);double total=0;
    for(size_t i=0;i<size;++i){output[i]=std::exp(input[i]-maximum);total+=output[i];}
    for(size_t i=0;i<size;++i)output[i]=float(output[i]/total);
}
void check(size_t tokens,size_t heads,size_t dim,size_t windows,bool quantized,bool sliced=false) {
    const size_t rows=tokens*windows,channels=heads*dim;std::mt19937 rng(unsigned(tokens+heads*17+dim*43));
    std::vector<float> input(rows*channels*3),bias(heads*tokens*tokens),actual(rows*channels),reference(actual.size());
    std::vector<int8_t> bytes(input.size());
    for(size_t i=0;i<input.size();++i){bytes[i]=int8_t(int(rng()%127)-63);input[i]=float(bytes[i])*0.03125f;}
    for(auto& x:bias)x=float(int(rng()%127)-63)*0.015625f;
    const auto read=[&](size_t row,size_t head,size_t part,size_t d){const size_t channel=(head*3+part)*dim+d;return input[(channel/4*rows+row)*4+channel%4];};
    std::vector<float> scratch((tokens+8)*(3*dim+4));
    for(size_t window=0;window<windows;++window)for(size_t h=0;h<heads;++h) {
        float* q=scratch.data();float* k=q+(tokens+8)*dim;float* v=k+(tokens+8)*dim;float* scores=v+(tokens+8)*dim;
        const void* source=quantized?static_cast<const void*>(bytes.data()):static_cast<const void*>(input.data());
        if(sliced) {
            const size_t middle=std::min(tokens,((tokens+3)/4+1)/2*4);
            tinyvit_attention::head(source,quantized,0.03125f,bias.data()+h*tokens*tokens,1.0f/std::sqrt(float(dim)),rows,tokens,heads,dim,window,h,q,k,v,scores,actual.data(),softmax,0,middle);
            tinyvit_attention::head(source,quantized,0.03125f,bias.data()+h*tokens*tokens,1.0f/std::sqrt(float(dim)),rows,tokens,heads,dim,window,h,q,k,v,scores,actual.data(),softmax,middle,tokens);
        } else tinyvit_attention::head(source,quantized,0.03125f,bias.data()+h*tokens*tokens,1.0f/std::sqrt(float(dim)),rows,tokens,heads,dim,window,h,q,k,v,scores,actual.data(),softmax);
        std::vector<double> probabilities(tokens);
        for(size_t i=0;i<tokens;++i) {
            double maximum=-1e300;
            for(size_t j=0;j<tokens;++j) {
                double dot=0;for(size_t d=0;d<dim;++d)dot+=double(read(window*tokens+i,h,0,d))*read(window*tokens+j,h,1,d);
                probabilities[j]=dot/std::sqrt(double(dim))+bias[(h*tokens+i)*tokens+j];maximum=std::max(maximum,probabilities[j]);
            }
            double total=0;for(auto& p:probabilities){p=std::exp(p-maximum);total+=p;}
            for(size_t d=0;d<dim;++d) {
                double sum=0;for(size_t j=0;j<tokens;++j)sum+=probabilities[j]/total*read(window*tokens+j,h,2,d);
                const size_t channel=h*dim+d;reference[(channel/4*rows+window*tokens+i)*4+channel%4]=float(sum);
            }
        }
    }
    double maximum=0;
    for(size_t i=0;i<actual.size();++i){const double error=std::abs(double(actual[i])-reference[i]);maximum=std::max(maximum,error);if(!std::isfinite(actual[i]) || error>1e-4+1e-4*std::abs(reference[i]))throw std::runtime_error("Fused attention oracle mismatch");}
    std::cout<<"PASS tokens="<<tokens<<" heads="<<heads<<" dim="<<dim<<" windows="<<windows<<" int8="<<quantized<<" sliced="<<sliced<<" outputs="<<actual.size()<<" maximum="<<maximum<<'\n';
}
}
int main(){try {
    for(bool quantized:{false,true})for(size_t tokens:{size_t(1),size_t(7),size_t(49),size_t(196)})check(tokens,4,32,2,quantized);
    check(49,10,32,1,false);check(196,5,32,1,true);check(7,1,4,3,false);check(1,1,8,1,true);
    for(bool quantized:{false,true})for(size_t tokens:{size_t(1),size_t(49),size_t(196)})check(tokens,5,32,1,quantized,true);
    std::cout<<"PASS 18 fused attention scalar-oracle cases\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
