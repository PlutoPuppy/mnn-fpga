/* DE1-SoC bring-up helper. Never changes SDRAM geometry, timing, staticcfg,
 * FPGA port widths/maps, or clocks. Quiesce only with no active FPGA users.
 * Registers: Linux altera-fpga2sdram.c and U-Boot socfpga reset/sdram headers.
 */
#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#if !defined(__arm__)
#error "This helper is only for the Cyclone V ARM HPS"
#endif

static void barrier(void) { __asm__ volatile("dmb sy" ::: "memory"); }
static uint32_t rd(volatile uint32_t *p,unsigned offset) {
    barrier();uint32_t value=p[offset/4];barrier();return value;
}
static void wr(volatile uint32_t *p,unsigned offset,uint32_t value) {
    barrier();p[offset/4]=value;barrier();
}
static void status(volatile uint32_t *sdr,volatile uint32_t *reset) {
    static const unsigned offsets[]={0,0x2c,0x30,0x5c,0x60,0x64,0x68,0x6c,0x70,0x74,0x78,0x7c,0x80};
    for(unsigned i=0;i<sizeof(offsets)/sizeof(offsets[0]);++i)
        printf("HPS_REG %08x=%08x\n",0xffc25000u+offsets[i],rd(sdr,offsets[i]));
    printf("HPS_REG ffd0501c=%08x\n",rd(reset,0x1c));
}
int main(int argc,char **argv) {
    const int inspect=argc==2 && strcmp(argv[1],"status")==0;
    const int quiesce=argc==3 && strcmp(argv[1],"quiesce")==0 && strcmp(argv[2],"--confirm-idle")==0;
    const int enable=argc==3 && strcmp(argv[1],"enable-mmio")==0 && strcmp(argv[2],"--confirm-idle")==0;
    if(!inspect && !quiesce && !enable) {
        fprintf(stderr,"Usage: socfpga_bridge_control status | quiesce --confirm-idle | enable-mmio --confirm-idle\n");return 2;
    }
    int fd=open("/dev/mem",(inspect ? O_RDONLY:O_RDWR)|O_SYNC);
    if(fd<0) {perror("open /dev/mem");return 1;}
    const int prot=PROT_READ|(inspect ? 0:PROT_WRITE);
    volatile uint32_t *sdr=mmap(NULL,4096,prot,MAP_SHARED,fd,(off_t)UINT64_C(0xffc25000));
    volatile uint32_t *reset=mmap(NULL,4096,prot,MAP_SHARED,fd,(off_t)UINT64_C(0xffd05000));
    if(sdr==MAP_FAILED || reset==MAP_FAILED) {perror("mmap HPS control");close(fd);return 1;}
    if(quiesce) {
        /* Isolate FPGA DDR ports, then assert H2F/LWH2F/F2H bridge resets. */
        wr(sdr,0x80,rd(sdr,0x80)&~UINT32_C(0x3fff));
        wr(reset,0x1c,rd(reset,0x1c)|7u);
    } else if(enable) {
        /* This PIO prototype needs no FPGA masters or FPGA DDR ports. */
        if(rd(sdr,0x80)&0x3fff) {
            fprintf(stderr,"FPGA DDR ports must remain disabled for PIO bring-up\n");return 1;
        }
        wr(reset,0x1c,(rd(reset,0x1c)|4u)&~3u);
    }
    status(sdr,reset);
    munmap((void*)sdr,4096);munmap((void*)reset,4096);close(fd);
    return 0;
}
