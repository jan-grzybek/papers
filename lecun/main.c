#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

// turns out MNIST is big-endian
// https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void read_error() {
    printf("Error reading file!");
    exit(1);
}

void malloc_error() {
    printf("Error allocating memory!");
    exit(1);
}

bool is_verbose() {
    if (getenv("VERBOSE") != NULL) {
        return strcmp(getenv("VERBOSE"), "1") == 0;
    } else return false;
}

void print_maybe(const char *format, ...) {
    static char verbose = -1;
    if (verbose == -1) {
        if (is_verbose()) verbose = 1;
        else verbose = 0;
    }
    if (verbose == 1) {
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
    }
}

typedef struct Images {
    int32_t magic_number;
    int32_t count;
    int32_t width;
    int32_t height;
    unsigned char* data;
} Images;

unsigned long images_allocate(Images* images) {
    print_maybe("Success");
    unsigned long memory_needed = images->count * images->width * images->height * sizeof(unsigned char);
    print_maybe("Success");
    images->data = malloc(memory_needed);
    print_maybe("Success");
    return memory_needed;
}

void images_deallocate(Images* images) {
    free(images->data);
}

Images rescale(Images images) {
    Images rescaled_images;
    rescaled_images.magic_number = images.magic_number;
    rescaled_images.count = images.count;
    rescaled_images.width = 16;
    rescaled_images.height = 16;
    images_allocate(&rescaled_images);

    int original_resolution = images.height * images.width;
    if (is_verbose()) {
        for (int img = 0; img < images.count; img++) {
            bool do_print = img % 10000 == 0;
            if (do_print) print_maybe("Image %d\n", img);
            else continue;
            for (int row = 0; row < images.height; row++) {
                for (int col = 0; col < images.width; col++) {
                    int idx = img * original_resolution + row * images.width + col;
                    if (images.data[idx] > 200) print_maybe("@@@");
                    else if (images.data[idx] > 100) print_maybe("111");
                    else if (images.data[idx] > 50) print_maybe("...");
                    else if (images.data[idx] > 25) print_maybe(" . ");
                    else print_maybe("   ");
                }
                print_maybe("\n");
            }
            print_maybe("\n");
        }
    }

    double xt_centers[rescaled_images.width];
    double x0_ratios[rescaled_images.width];
    double x1_ratios[rescaled_images.width];
    int x0_indices[rescaled_images.width];
    int x1_indices[rescaled_images.width];
    double yt_centers[rescaled_images.height];
    double y0_ratios[rescaled_images.height];
    double y1_ratios[rescaled_images.height];
    int y0_indices[rescaled_images.height];
    int y1_indices[rescaled_images.height];
    for (int i = 0; i < rescaled_images.width; i++) {
        xt_centers[i] = (2 * (double)i + 1) * (double)images.width / (2 * (double)rescaled_images.width);
        x0_indices[i] = (int)(xt_centers[i] - 0.5);
        int x1 = (int)(xt_centers[i] + 0.5);
        x0_ratios[i] = x1 - xt_centers[i] + 0.5;
        x1_ratios[i] = xt_centers[i] - (x0_indices[i] + 0.5);
        if (x1 < images.width) x1_indices[i] = x1;
        else x1_indices[i] = images.width - 1;
    }
    for (int i = 0; i < rescaled_images.height; i++) {
        yt_centers[i] = (2 * (double)i + 1) * (double)images.height / (2 * (double)rescaled_images.height);
        y0_indices[i] = (int)(yt_centers[i] - 0.5);
        int y1 = (int)(yt_centers[i] + 0.5);
        y0_ratios[i] = y1 - yt_centers[i] + 0.5;
        y1_ratios[i] = yt_centers[i] - (y0_indices[i] + 0.5);
        if (y1 < images.height) y1_indices[i] = y1;
        else y1_indices[i] = images.height - 1;
    }

//    for (int i = 0; i < rescaled_images.width; i++) {
//        printf("------\n");
//        printf("%f\n", xt_centers[i]);
//        printf("%f\n", x0_ratios[i]);
//        printf("%f\n", x1_ratios[i]);
//        printf("%d\n", x0_indices[i]);
//        printf("%d\n", x1_indices[i]);
//    }

    int target_resolution = rescaled_images.height * rescaled_images.width;
    for (int img = 0; img < rescaled_images.count; img++) {
        int idx0 =  img * target_resolution;
        int img_offset = img * original_resolution;
        for (int row = 0; row < rescaled_images.height; row++) {
            int idx1 = idx0 + row * rescaled_images.width;
            int y0_offset = y0_indices[row] * images.width + img_offset;
            int y1_offset = y1_indices[row] * images.width + img_offset;
            for (int col = 0; col < rescaled_images.width; col++) {
                double a = x0_ratios[col]*images.data[x0_indices[col]+y0_offset];
                double b = x1_ratios[col]*images.data[x1_indices[col]+y0_offset];
                double c = x0_ratios[col]*images.data[x0_indices[col]+y1_offset];
                double d = x1_ratios[col]*images.data[x1_indices[col]+y1_offset];
                rescaled_images.data[idx1 + col] = (unsigned char)(y0_ratios[row] * (a+b) + y1_ratios[row] * (c+d));
            }
        }
    }

    if (is_verbose()) {
        for (int img = 0; img < rescaled_images.count; img++) {
            bool do_print = img % 10000 == 0;
            if (do_print) print_maybe("Image %d\n", img);
            else continue;
            for (int row = 0; row < rescaled_images.height; row++) {
                for (int col = 0; col < rescaled_images.width; col++) {
                    int idx = img * target_resolution + row * rescaled_images.width + col;
                    if (rescaled_images.data[idx] > 200) print_maybe("@@@");
                    else if (rescaled_images.data[idx] > 100) print_maybe("111");
                    else if (rescaled_images.data[idx] > 50) print_maybe("...");
                    else if (rescaled_images.data[idx] > 25) print_maybe(" . ");
                    else print_maybe("   ");
                }
                print_maybe("\n");
            }
            print_maybe("\n");
        }
    }

    images_deallocate(&images);
    return rescaled_images;
}

Images load_mnist() {
    FILE *fptr;
    fptr = fopen("/Users/jan/Downloads/train-images-idx3-ubyte", "rb");
    if(fptr == NULL) read_error();

    Images images;
    fread(&images, sizeof(int32_t), 4, fptr);
    bool swap_endianness = images.magic_number != 2051;
    if (swap_endianness) images.magic_number = reverseInt(images.magic_number);
    if (images.magic_number != 2051) read_error();
    if (swap_endianness) {
        images.count = reverseInt(images.count);
        images.width = reverseInt(images.width);
        images.height = reverseInt(images.height);
    }

    int offset = 4 * sizeof(int32_t);
    fseek(fptr, offset, SEEK_SET);

    print_maybe("Loading %d images ...\n", images.count);
    unsigned long memory_needed = images_allocate(&images);
    fread(images.data, memory_needed, 1, fptr);

    fclose(fptr);
    print_maybe("Success.");
    return images;
}

int main() {
    rescale(load_mnist());
    return 0;
}
