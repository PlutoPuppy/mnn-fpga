#include "MNN_generated.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
std::unique_ptr<MNN::NetT> load(const char* p){std::ifstream in(p,std::ios::binary);if(!in)throw std::runtime_error("No input");std::vector<uint8_t>b((std::istreambuf_iterator<char>(in)),{});flatbuffers::Verifier v(b.data(),b.size());if(!MNN::VerifyNetBuffer(v))throw std::runtime_error("Invalid model");return std::unique_ptr<MNN::NetT>(MNN::GetNet(b.data())->UnPack());}
int main(int n,char** a){try{if(n!=5)throw std::runtime_error("FP32 CALIBRATED ALLOWLIST NEW_OUTPUT");if(std::ifstream(a[4]))throw std::runtime_error("Output exists");auto f=load(a[1]),q=load(a[2]);if(f->tensorName!=q->tensorName||f->oplists.size()!=q->oplists.size())throw std::runtime_error("Graph mismatch");std::set<std::string>allow,done;std::ifstream policy(a[3]);std::string name;while(std::getline(policy,name)){if(!name.empty()&&name.back()=='\r')name.pop_back();if(!name.empty()&&name[0]!='#')allow.insert(name);}for(size_t i=0;i<f->oplists.size();++i){auto& op=f->oplists[i];auto& src=q->oplists[i];if(op->name!=src->name)throw std::runtime_error("Op mismatch");if(!allow.count(op->name))continue;auto* c=op->main.AsConvolution2D();auto* qc=src->main.AsConvolution2D();if(!c||!qc||!qc->quanParameter)throw std::runtime_error("Missing quant weights");c->quanParameter.reset(new MNN::IDSTQuanT(*qc->quanParameter));c->quanParameter->has_scaleInt=false;c->weight.clear();done.insert(op->name);}if(done!=allow)throw std::runtime_error("Missing names");flatbuffers::FlatBufferBuilder b;b.Finish(MNN::Net::Pack(b,f.get()));std::ofstream out(a[4],std::ios::binary);out.write(reinterpret_cast<const char*>(b.GetBufferPointer()),b.GetSize());if(!out)throw std::runtime_error("Write failed");std::cout<<"Weight-only "<<done.size()<<" bytes "<<b.GetSize()<<'\n';}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
