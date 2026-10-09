// Dedicated non-DMA, polled UIO platform driver. Build against board kernel.
#include <linux/module.h>
#include <linux/mm.h>
#include <linux/ioport.h>
#include <linux/of.h>
#include <linux/platform_device.h>
#include <linux/slab.h>
#include <linux/uio_driver.h>

static int systolic_probe(struct platform_device *pdev)
{
    struct resource *res;
    struct uio_info *info;

    res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
    if (!res || resource_size(res) != 0x1000 || (res->start & (PAGE_SIZE - 1)))
        return -EINVAL;
    if (!devm_request_mem_region(&pdev->dev, res->start, resource_size(res),
                                 dev_name(&pdev->dev)))
        return -EBUSY;
    info = devm_kzalloc(&pdev->dev, sizeof(*info), GFP_KERNEL);
    if (!info)
        return -ENOMEM;
    info->name = "systolic-mmio";
    info->version = "1.0";
    info->irq = UIO_IRQ_NONE;
    info->mem[0].name = "registers-and-tiles";
    info->mem[0].addr = res->start;
    info->mem[0].size = resource_size(res);
    info->mem[0].memtype = UIO_MEM_PHYS;
    return devm_uio_register_device(&pdev->dev, info);
}

static const struct of_device_id systolic_match[] = {
    { .compatible = "mnn,systolic-mmio-1.0" },
    { }
};
MODULE_DEVICE_TABLE(of, systolic_match);
static struct platform_driver systolic_driver = {
    .probe = systolic_probe,
    .driver = {
        .name = "systolic-mmio",
        .of_match_table = systolic_match,
    },
};
module_platform_driver(systolic_driver);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Polled UIO driver for 16x16 systolic GEMM");
