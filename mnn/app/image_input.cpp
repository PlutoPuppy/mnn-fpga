#include "image_input.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

std::vector<float> image_input(const std::string& path, int width, int height,
    int resize_short, const std::array<float,3>& mean, const std::array<float,3>& stddev) {
    int iw=0, ih=0, channels=0;
    std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels(
        stbi_load(path.c_str(), &iw, &ih, &channels, 3), stbi_image_free);
    if(!pixels) throw std::runtime_error(std::string("Image decode failed: ") + stbi_failure_reason());
    int rw=width, rh=height;
    if(resize_short) {
        const double scale=double(resize_short)/std::min(iw,ih);
        const double scaled_w=std::floor(iw*scale), scaled_h=std::floor(ih*scale);
        if(scaled_w>100000 || scaled_h>100000) throw std::runtime_error("Resized image dimensions too large");
        rw=int(scaled_w); rh=int(scaled_h);
        if(rw<width || rh<height) throw std::runtime_error("--resize-short is too small for the requested crop");
    }
    const int crop_x=(rw-width)/2, crop_y=(rh-height)/2;
    std::vector<float> out(size_t(3)*width*height);
    // Bilinear resize at pixel centers; clamp border samples. Crop is centered
    // in the resized image. No antialias filter, EXIF rotation, or color management.
    for(int y=0;y<height;++y) for(int x=0;x<width;++x) {
        const float sx=std::clamp(float((x+crop_x+0.5)*iw/rw-0.5),0.0f,float(iw-1));
        const float sy=std::clamp(float((y+crop_y+0.5)*ih/rh-0.5),0.0f,float(ih-1));
        const int x0=int(sx), y0=int(sy), x1=std::min(x0+1,iw-1), y1=std::min(y0+1,ih-1);
        const float dx=sx-x0, dy=sy-y0;
        for(int c=0;c<3;++c) {
            auto at=[&](int xx,int yy) {return float(pixels.get()[(size_t(yy)*iw+xx)*3+c]);};
            const float v=(1-dy)*((1-dx)*at(x0,y0)+dx*at(x1,y0))+
                               dy*((1-dx)*at(x0,y1)+dx*at(x1,y1));
            out[(size_t(c)*height+y)*width+x]=(v/255.0f-mean[c])/stddev[c];
        }
    }
    return out;
}
