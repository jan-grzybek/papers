#include <stdio.h>
#include <stdlib.h>

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

    printf("%d\n", meta_data.height);
    unsigned char ***images = (unsigned char ***)malloc(meta_data.images_count * sizeof(unsigned char **));
    if (images == NULL) malloc_error();
    for (int idx = 0; idx < meta_data.images_count; idx++) {
        images[idx] = (unsigned char **)malloc(meta_data.height * sizeof(unsigned char *));
        if (images[idx] == NULL) malloc_error();
        for (int row = 0; row < meta_data.height; row++) {
            images[idx][row] = (unsigned char *)malloc(meta_data.width * sizeof(unsigned char));
            if (images[idx][row] == NULL) malloc_error();
        }
    }
    int offset = 4 * sizeof(int32_t);
    fseek(fptr, offset, SEEK_SET);

    for (int idx = 0; idx < meta_data.images_count; idx++) {
        for (int row = 0; row < meta_data.height; row++) {
            for (int col = 0; col < meta_data.width; col++) {
                fread(&images[idx][row][col], sizeof(unsigned char), 1, fptr);
                offset++;
                fseek(fptr, offset, SEEK_SET);
                if (images[idx][row][col] > 200) printf("@");
                else if (images[idx][row][col] > 100) printf("1");
                else if (images[idx][row][col] > 50) printf(".");
                else printf(" ");
            }
            printf("\n");
        }
        printf("\n");
    }

    fclose(fptr);
    return 0;
}

int main() {
    load_mnist();
    return 0;
}
