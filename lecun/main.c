#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

// turns out MNIST is big-endian
// https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
int reverseInt(int i)
{
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

struct ImagesMetaData {
    int32_t magic_number;
    int32_t images_count;
    int32_t width;
    int32_t height;
};

int load_mnist() {
    FILE *fptr;
    fptr = fopen("/Users/jan/Downloads/train-images-idx3-ubyte", "rb");
    if(fptr == NULL) read_error();

    struct ImagesMetaData meta_data;
    fread(&meta_data, sizeof(int32_t), 4, fptr);
    bool swap_endianness = meta_data.magic_number != 2051;
    if (swap_endianness) meta_data.magic_number = reverseInt(meta_data.magic_number);
    if (meta_data.magic_number != 2051) read_error();
    if (swap_endianness) {
        meta_data.images_count = reverseInt(meta_data.images_count);
        meta_data.width = reverseInt(meta_data.width);
        meta_data.height = reverseInt(meta_data.height);
    }

    unsigned char (*images)[meta_data.images_count][meta_data.height][meta_data.width] = malloc(
            sizeof(unsigned char[meta_data.images_count][meta_data.height][meta_data.width]));
    int offset = 4 * sizeof(int32_t);
    fseek(fptr, offset, SEEK_SET);

    print_maybe("Loading %d images ...\n", meta_data.images_count);
    fread(images, sizeof(
            unsigned char[meta_data.images_count][meta_data.height][meta_data.width]), 1, fptr);

    if (is_verbose()) {
        for (int idx = 0; idx < meta_data.images_count; idx++) {
            bool do_print = idx % 10000 == 0;
            if (do_print) print_maybe("Image %d\n", idx);
            for (int row = 0; row < meta_data.height; row++) {
                for (int col = 0; col < meta_data.width; col++) {
                    if (do_print) {
                        if ((*images)[idx][row][col] > 200) print_maybe("@@@");
                        else if ((*images)[idx][row][col] > 100) print_maybe("111");
                        else if ((*images)[idx][row][col] > 50) print_maybe("...");
                        else print_maybe("   ");
                    }
                }
                if (do_print) print_maybe("\n");
            }
            if (do_print) print_maybe("\n");
        }
    }

    fclose(fptr);
    return 0;
}

int main() {
    load_mnist();
    return 0;
}
