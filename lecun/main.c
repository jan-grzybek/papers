#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>


const int example_idx = 47;

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

void error_out(char *error_log) {
    printf("Error: ");
    printf("%s", error_log);
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

// inspired by https://stackoverflow.com/questions/6127503/shuffle-array-in-c
void shuffle_int_array(int *array, size_t n) {
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
            double z = (double)rand() / RAND_MAX;
            size_t j = i + (int)(z * (double)(n - i));
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

typedef struct Labels {
    int32_t magic_number;
    int32_t count;
    unsigned char *data;
} Labels;

unsigned long labels_allocate(Labels *labels) {
    unsigned long memory_needed = labels->count * sizeof(unsigned char);
    labels->data = malloc(memory_needed);
    return memory_needed;
}

void labels_deallocate(Labels *labels) {
    free(labels->data);
    labels->data = NULL;
}

typedef struct Images {
    int magic_number, count, height, width, size;
    unsigned char *data_byte;
    float *data_float;
} Images;

unsigned long images_allocate_byte(Images *images) {
    unsigned long memory_needed = images->count * images->size * sizeof(unsigned char);
    images->data_byte = malloc(memory_needed);
    return memory_needed;
}

void images_deallocate_byte(Images *images) {
    free(images->data_byte);
    images->data_byte = NULL;
}

unsigned long images_allocate_float(Images *images) {
    unsigned long memory_needed = images->count * images->size * sizeof(float);
    images->data_float = malloc(memory_needed);
    return memory_needed;
}

void images_deallocate_float(Images *images) {
    free(images->data_float);
    images->data_float = NULL;
}

void show_image(Images images, int index) {
    int img_offset = index * images.size;
    print_maybe("\nImage %d, %d x %d\n", index, images.height, images.width);
    for (int col = 0; col < images.width+2; col++) print_maybe("# #"); print_maybe("\n");
    for (int row = 0; row < images.height; row++) {
        print_maybe("#  ");
        int row_offset = img_offset + row * images.width;
        for (int col = 0; col < images.width; col++) {
            int idx = row_offset + col;
            if (images.data_byte[idx] > 200) print_maybe("@@@");
            else if (images.data_byte[idx] > 100) print_maybe("111");
            else if (images.data_byte[idx] > 50) print_maybe("...");
            else if (images.data_byte[idx] > 25) print_maybe(" . ");
            else print_maybe("   ");
        }
        print_maybe("  #");
        print_maybe("\n");
    }
    for (int col = 0; col < images.width+2; col++) print_maybe("# #"); print_maybe("\n");
}

struct BilinearGrid {
    double* target_centers;
    double* ratios_first;
    double* ratios_second;
    int* indices_first;
    int* indices_second;
};

void grid_allocate(struct BilinearGrid *grid, int target_size) {
    unsigned long memory_needed_double = target_size * sizeof(double);
    unsigned long memory_needed_int = target_size * sizeof(int);
    grid->target_centers = malloc(memory_needed_double);
    grid->ratios_first = malloc(memory_needed_double);
    grid->ratios_second = malloc(memory_needed_double);
    grid->indices_first = malloc(memory_needed_int);
    grid->indices_second = malloc(memory_needed_int);
}

void grid_deallocate(struct BilinearGrid *grid) {
    free(grid->target_centers);
    free(grid->ratios_first);
    free(grid->ratios_second);
    free(grid->indices_first);
    free(grid->indices_second);
    grid->target_centers = NULL;
    grid->ratios_first = NULL;
    grid->ratios_second = NULL;
    grid->indices_first = NULL;
    grid->indices_second = NULL;
}

void grid_calculate(struct BilinearGrid *grid, int original_size, int target_size) {
    for (int i = 0; i < target_size; i++) {
        grid->target_centers[i] = (2 * (double)i + 1) * (double)original_size / (2 * (double)target_size);
        grid->indices_first[i] = (int)(grid->target_centers[i] - 0.5);
        int x1 = (int)(grid->target_centers[i] + 0.5);
        grid->ratios_first[i] = x1 - grid->target_centers[i] + 0.5;
        grid->ratios_second[i] = grid->target_centers[i] - (grid->indices_first[i] + 0.5);
        if (x1 < original_size) grid->indices_second[i] = x1;
        else grid->indices_second[i] = original_size - 1;
    }
}

Images resize_bilinear(Images images, int target_width, int target_height) {
    Images rescaled_images;
    rescaled_images.magic_number = images.magic_number;
    rescaled_images.count = images.count;
    rescaled_images.width = target_width;
    rescaled_images.height = target_height;
    rescaled_images.size = target_height * target_width;
    images_allocate_byte(&rescaled_images);

    struct BilinearGrid horizontal_grid;
    grid_allocate(&horizontal_grid, rescaled_images.width);
    grid_calculate(&horizontal_grid, images.width, rescaled_images.width);
    struct BilinearGrid vertical_grid;
    grid_allocate(&vertical_grid, rescaled_images.height);
    grid_calculate(&vertical_grid, images.height, rescaled_images.height);

    int original_resolution = images.height * images.width;
    int target_resolution = rescaled_images.height * rescaled_images.width;
    for (int img = 0; img < rescaled_images.count; img++) {
        int t_img_offset =  img * target_resolution;
        int o_img_offset = img * original_resolution;
        for (int row = 0; row < rescaled_images.height; row++) {
            int idx1 = t_img_offset + row * rescaled_images.width;
            int y0_offset = vertical_grid.indices_first[row] * images.width + o_img_offset;
            int y1_offset = vertical_grid.indices_second[row] * images.width + o_img_offset;
            for (int col = 0; col < rescaled_images.width; col++) {
                double a = horizontal_grid.ratios_first[col] *
                        images.data_byte[horizontal_grid.indices_first[col]+y0_offset];
                double b = horizontal_grid.ratios_second[col] *
                        images.data_byte[horizontal_grid.indices_second[col]+y0_offset];
                double c = horizontal_grid.ratios_first[col] *
                        images.data_byte[horizontal_grid.indices_first[col]+y1_offset];
                double d = horizontal_grid.ratios_second[col] *
                        images.data_byte[horizontal_grid.indices_second[col]+y1_offset];
                rescaled_images.data_byte[idx1 + col] = (unsigned char)(vertical_grid.ratios_first[row] * (a+b) +
                        vertical_grid.ratios_second[row] * (c+d));
            }
        }
    }

    grid_deallocate(&horizontal_grid);
    grid_deallocate(&vertical_grid);

    if (is_verbose()) {
        show_image(images, example_idx);
        show_image(rescaled_images, example_idx);
    }

    images_deallocate_byte(&images);
    return rescaled_images;
}

Images normalize(Images images) {
    images_allocate_float(&images);
    for (int img = 0; img < images.count; img++) {
        int img_offset = img * images.size;
        for (int row = 0; row < images.height; row++) {
            int row_offset = img_offset + row * images.width;
            for (int col = 0; col < images.width; col++) {
                int offset = row_offset + col;
                images.data_float[offset] = ((float)images.data_byte[offset] / (float)127.5) - 1;
            }
        }
    }
    images_deallocate_byte(&images);
    return images;
}

Labels mnist_load_labels(char *filepath, int magic_number) {
    FILE *fptr;
    fptr = fopen(filepath, "rb");
    if(fptr == NULL) error_out("Can't read file.");

    Labels labels;
    fread(&labels, sizeof(int32_t), 2, fptr);
    bool swap_endianness = labels.magic_number != magic_number;
    if (swap_endianness) labels.magic_number = reverseInt(labels.magic_number);
    if (labels.magic_number != magic_number) error_out("Magic number mismatch.");
    if (swap_endianness) {
        labels.count = reverseInt(labels.count);
    }

    int offset = 2 * sizeof(int32_t);
    fseek(fptr, offset, SEEK_SET);

    print_maybe("\nLoading %d labels ... ", labels.count);
    unsigned long memory_needed = labels_allocate(&labels);
    fread(labels.data, memory_needed, 1, fptr);

    fclose(fptr);
    print_maybe("success.\n");
    print_maybe("Label %d: %d\n", example_idx, labels.data[example_idx]);
    return labels;
}

Images mnist_load_images(char *filepath, int magic_number) {
    FILE *fptr;
    fptr = fopen(filepath, "rb");
    if(fptr == NULL) error_out("Can't read file.");

    Images images;
    fread(&images, sizeof(int32_t), 4, fptr);
    bool swap_endianness = images.magic_number != magic_number;
    if (swap_endianness) images.magic_number = reverseInt(images.magic_number);
    if (images.magic_number != magic_number) error_out("Magic number mismatch.");
    if (swap_endianness) {
        images.count = reverseInt(images.count);
        images.width = reverseInt(images.width);
        images.height = reverseInt(images.height);
    }
    images.size = images.width * images.height;

    int offset = 4 * sizeof(int32_t);
    fseek(fptr, offset, SEEK_SET);

    print_maybe("Loading %d images ... ", images.count);
    unsigned long memory_needed = images_allocate_byte(&images);
    fread(images.data_byte, memory_needed, 1, fptr);

    fclose(fptr);
    print_maybe("success.\n");
    return images;
}

typedef struct Indices {
    int *train_set;
    int *test_set;
} Indices;

void indices_allocate(Indices *indices, int train_count, int test_count) {
    indices->train_set = malloc(train_count * sizeof(int));
    indices->test_set = malloc(test_count * sizeof(int));
}

void indices_deallocate(Indices *indices) {
    free(indices->train_set);
    indices->train_set = NULL;
    free(indices->test_set);
    indices->test_set = NULL;
}

Indices split_dataset(int total_samples, int train_samples, int test_samples) {
    int indices[total_samples];
    for (int i = 0; i < total_samples; i++) indices[i] = i;
    shuffle_int_array(indices, total_samples);
    Indices indices_split;
    indices_allocate(&indices_split, train_samples, test_samples);
    for (int i = 0; i < train_samples; i++) indices_split.train_set[i] = indices[i];
    for (int i = 0; i < test_samples; i++) indices_split.test_set[i] = indices[i+train_samples];
    return indices_split;
}

typedef struct Data {
    int *dims;
    float *data;
} Data;

void data_deallocate(Data *data) {
//    free(data->dims);
//    data->dims = NULL;
    free(data->data);
    data->data = NULL;
}

typedef struct Conv2D {
    int stride, height, width;
    bool padding;
    Data output;
    float *weights;
    void (*forward)(struct Conv2D*, Data*);
    void (*reset_output)(struct Conv2D*, Data*);
} Conv2D;

void fill_array_random_floats(float *array, int n, double range_start, double range_end) {
    if (range_end < range_start) error_out("range_end smaller than range_start");
    double range = range_end - range_start;
    for (int idx = 0; idx < n; idx++) array[idx] = (float)(range * rand() / RAND_MAX + range_start);
}

void Conv2DResetOutput(Conv2D *layer, Data *input) {
    if (layer->output.data != NULL) data_deallocate(&layer->output);
    if (layer->padding) {
        int out_h = (input->dims[0] - 1) / layer->stride + 1;
        int out_w = (input->dims[1] - 1) / layer->stride + 1;
        layer->output.dims = (int[2]) {out_h, out_w};
    } else {
        if (input->dims[0] < layer->height) error_out("input dim [0] smaller than kernel height");
        if (input->dims[1] < layer->width) error_out("input dim [1] smaller than kernel width");
        int out_h = (input->dims[0] - layer->height) / layer->stride + 1;
        int out_w = (input->dims[1] - layer->width) / layer->stride + 1;
        layer->output.dims = (int[2]) {out_h, out_w};
    }
    int output_size = layer->output.dims[0] * layer->output.dims[1];
    layer->output.data = malloc(output_size * sizeof(float));
    for (int i = 0; i < output_size; i++) layer->output.data[i] = 0;
}

void Conv2DForward(Conv2D *layer, Data *input) {
    int x_start = (layer->padding) ? -layer->width / 2 : 0;
    int y_start = (layer->padding) ? -layer->height / 2 : 0;
    int out_idx = 0;
    for (int y = y_start; y < y_start + layer->output.dims[0] * layer->stride; y += layer->stride) {
        for (int x = x_start; x < x_start + layer->output.dims[1] * layer->stride; x += layer->stride) {
            float out = 0;
            for (int yk = 0; yk < layer->height; yk++) {
                for (int xk = 0; xk < layer->width; xk++) {
                    int x_shifted = x + xk;
                    int y_shifted = y + yk;
                    if (x_shifted < 0 || y_shifted < 0 || input->dims[1] <= x_shifted || input->dims[0] <= y_shifted)
                        out -= layer->weights[yk * layer->width + xk];
                    else out += layer->weights[yk * layer->width + xk] *
                            input->data[y_shifted * input->dims[1] + x_shifted];
                }
            }
            layer->output.data[out_idx] += out;
            out_idx++;
        }
    }
}

Conv2D Conv2DInit(int stride, bool padding, int height, int width) {
    if (stride < 1) error_out("Conv2D stride has to be >= 1");
    if (height < 1) error_out("kernel's height has to be >= 1");
    if (width < 1) error_out("kernel's width has to be >= 1");
    Conv2D conv;
    conv.forward = Conv2DForward;
    conv.reset_output = Conv2DResetOutput;
    conv.stride = stride;
    conv.padding = padding;
    conv.height = height;
    conv.width = width;
    int kernel_size = height * width;
    conv.weights = malloc(kernel_size * sizeof(float));
    conv.output.data = NULL;
    fill_array_random_floats(conv.weights, kernel_size,
                             -2.4/(double)kernel_size, 2.4/(double)kernel_size);
    // for (int i =0; i < 25; i++) conv.weights[i] = i % 2;
    return conv;
}

typedef struct LeNet {
    Conv2D H1_1, H1_2, H1_3, H1_4, H1_5, H1_6, H1_7, H1_8, H1_9, H1_10, H1_11, H1_12;
    Conv2D H2_1, H2_2, H2_3, H2_4, H2_5, H2_6, H2_7, H2_8, H2_9, H2_10, H2_11, H2_12;
    void (*forward)(struct LeNet*, Data*, Data*);
} LeNet;

void LeNetForward(LeNet *lenet, Data *input, Data *output) {
    lenet->H1_1.reset_output(&lenet->H1_1, input);
    lenet->H1_2.reset_output(&lenet->H1_2, input);
    lenet->H1_3.reset_output(&lenet->H1_3, input);
    lenet->H1_4.reset_output(&lenet->H1_4, input);
    lenet->H1_5.reset_output(&lenet->H1_5, input);
    lenet->H1_6.reset_output(&lenet->H1_6, input);
    lenet->H1_7.reset_output(&lenet->H1_7, input);
    lenet->H1_8.reset_output(&lenet->H1_8, input);
    lenet->H1_9.reset_output(&lenet->H1_9, input);
    lenet->H1_10.reset_output(&lenet->H1_10, input);
    lenet->H1_11.reset_output(&lenet->H1_11, input);
    lenet->H1_12.reset_output(&lenet->H1_12, input);

    lenet->H1_1.forward(&lenet->H1_1, input);
    lenet->H1_2.forward(&lenet->H1_2, input);
    lenet->H1_3.forward(&lenet->H1_3, input);
    lenet->H1_4.forward(&lenet->H1_4, input);
    lenet->H1_5.forward(&lenet->H1_5, input);
    lenet->H1_6.forward(&lenet->H1_6, input);
    lenet->H1_7.forward(&lenet->H1_7, input);
    lenet->H1_8.forward(&lenet->H1_8, input);
    lenet->H1_9.forward(&lenet->H1_9, input);
    lenet->H1_10.forward(&lenet->H1_10, input);
    lenet->H1_11.forward(&lenet->H1_11, input);
    lenet->H1_12.forward(&lenet->H1_12, input);

    lenet->H2_1.reset_output(&lenet->H2_1, &lenet->H1_1.output);
    lenet->H2_2.reset_output(&lenet->H2_2, &lenet->H1_1.output);
    lenet->H2_3.reset_output(&lenet->H2_3, &lenet->H1_1.output);
    lenet->H2_4.reset_output(&lenet->H2_4, &lenet->H1_1.output);
    lenet->H2_5.reset_output(&lenet->H2_5, &lenet->H1_1.output);
    lenet->H2_6.reset_output(&lenet->H2_6, &lenet->H1_1.output);
    lenet->H2_7.reset_output(&lenet->H2_7, &lenet->H1_1.output);
    lenet->H2_8.reset_output(&lenet->H2_8, &lenet->H1_1.output);
    lenet->H2_9.reset_output(&lenet->H2_9, &lenet->H1_1.output);
    lenet->H2_10.reset_output(&lenet->H2_10, &lenet->H1_1.output);
    lenet->H2_11.reset_output(&lenet->H2_11, &lenet->H1_1.output);
    lenet->H2_12.reset_output(&lenet->H2_12, &lenet->H1_1.output);

    lenet->H2_1.forward(&lenet->H2_1, &lenet->H1_1.output);
    lenet->H2_1.forward(&lenet->H2_1, &lenet->H1_2.output);
    lenet->H2_1.forward(&lenet->H2_1, &lenet->H1_4.output);
    lenet->H2_1.forward(&lenet->H2_1, &lenet->H1_5.output);
    lenet->H2_1.forward(&lenet->H2_1, &lenet->H1_7.output);
    lenet->H2_1.forward(&lenet->H2_1, &lenet->H1_8.output);
    lenet->H2_1.forward(&lenet->H2_1, &lenet->H1_10.output);
    lenet->H2_1.forward(&lenet->H2_1, &lenet->H1_11.output);

    lenet->H2_2.forward(&lenet->H2_2, &lenet->H1_1.output);
    lenet->H2_2.forward(&lenet->H2_2, &lenet->H1_2.output);
    lenet->H2_2.forward(&lenet->H2_2, &lenet->H1_4.output);
    lenet->H2_2.forward(&lenet->H2_2, &lenet->H1_5.output);
    lenet->H2_2.forward(&lenet->H2_2, &lenet->H1_7.output);
    lenet->H2_2.forward(&lenet->H2_2, &lenet->H1_8.output);
    lenet->H2_2.forward(&lenet->H2_2, &lenet->H1_10.output);
    lenet->H2_2.forward(&lenet->H2_2, &lenet->H1_11.output);

    lenet->H2_3.forward(&lenet->H2_3, &lenet->H1_1.output);
    lenet->H2_3.forward(&lenet->H2_3, &lenet->H1_2.output);
    lenet->H2_3.forward(&lenet->H2_3, &lenet->H1_4.output);
    lenet->H2_3.forward(&lenet->H2_3, &lenet->H1_5.output);
    lenet->H2_3.forward(&lenet->H2_3, &lenet->H1_7.output);
    lenet->H2_3.forward(&lenet->H2_3, &lenet->H1_8.output);
    lenet->H2_3.forward(&lenet->H2_3, &lenet->H1_10.output);
    lenet->H2_3.forward(&lenet->H2_3, &lenet->H1_11.output);

    lenet->H2_4.forward(&lenet->H2_4, &lenet->H1_1.output);
    lenet->H2_4.forward(&lenet->H2_4, &lenet->H1_2.output);
    lenet->H2_4.forward(&lenet->H2_4, &lenet->H1_4.output);
    lenet->H2_4.forward(&lenet->H2_4, &lenet->H1_5.output);
    lenet->H2_4.forward(&lenet->H2_4, &lenet->H1_7.output);
    lenet->H2_4.forward(&lenet->H2_4, &lenet->H1_8.output);
    lenet->H2_4.forward(&lenet->H2_4, &lenet->H1_10.output);
    lenet->H2_4.forward(&lenet->H2_4, &lenet->H1_11.output);

    lenet->H2_5.forward(&lenet->H2_5, &lenet->H1_1.output);
    lenet->H2_5.forward(&lenet->H2_5, &lenet->H1_3.output);
    lenet->H2_5.forward(&lenet->H2_5, &lenet->H1_4.output);
    lenet->H2_5.forward(&lenet->H2_5, &lenet->H1_6.output);
    lenet->H2_5.forward(&lenet->H2_5, &lenet->H1_7.output);
    lenet->H2_5.forward(&lenet->H2_5, &lenet->H1_9.output);
    lenet->H2_5.forward(&lenet->H2_5, &lenet->H1_10.output);
    lenet->H2_5.forward(&lenet->H2_5, &lenet->H1_12.output);

    lenet->H2_6.forward(&lenet->H2_6, &lenet->H1_1.output);
    lenet->H2_6.forward(&lenet->H2_6, &lenet->H1_3.output);
    lenet->H2_6.forward(&lenet->H2_6, &lenet->H1_4.output);
    lenet->H2_6.forward(&lenet->H2_6, &lenet->H1_6.output);
    lenet->H2_6.forward(&lenet->H2_6, &lenet->H1_7.output);
    lenet->H2_6.forward(&lenet->H2_6, &lenet->H1_9.output);
    lenet->H2_6.forward(&lenet->H2_6, &lenet->H1_10.output);
    lenet->H2_6.forward(&lenet->H2_6, &lenet->H1_12.output);

    lenet->H2_7.forward(&lenet->H2_7, &lenet->H1_1.output);
    lenet->H2_7.forward(&lenet->H2_7, &lenet->H1_3.output);
    lenet->H2_7.forward(&lenet->H2_7, &lenet->H1_4.output);
    lenet->H2_7.forward(&lenet->H2_7, &lenet->H1_6.output);
    lenet->H2_7.forward(&lenet->H2_7, &lenet->H1_7.output);
    lenet->H2_7.forward(&lenet->H2_7, &lenet->H1_9.output);
    lenet->H2_7.forward(&lenet->H2_7, &lenet->H1_10.output);
    lenet->H2_7.forward(&lenet->H2_7, &lenet->H1_12.output);

    lenet->H2_8.forward(&lenet->H2_8, &lenet->H1_1.output);
    lenet->H2_8.forward(&lenet->H2_8, &lenet->H1_3.output);
    lenet->H2_8.forward(&lenet->H2_8, &lenet->H1_4.output);
    lenet->H2_8.forward(&lenet->H2_8, &lenet->H1_6.output);
    lenet->H2_8.forward(&lenet->H2_8, &lenet->H1_7.output);
    lenet->H2_8.forward(&lenet->H2_8, &lenet->H1_9.output);
    lenet->H2_8.forward(&lenet->H2_8, &lenet->H1_10.output);
    lenet->H2_8.forward(&lenet->H2_8, &lenet->H1_12.output);

    lenet->H2_9.forward(&lenet->H2_9, &lenet->H1_2.output);
    lenet->H2_9.forward(&lenet->H2_9, &lenet->H1_3.output);
    lenet->H2_9.forward(&lenet->H2_9, &lenet->H1_5.output);
    lenet->H2_9.forward(&lenet->H2_9, &lenet->H1_6.output);
    lenet->H2_9.forward(&lenet->H2_9, &lenet->H1_8.output);
    lenet->H2_9.forward(&lenet->H2_9, &lenet->H1_9.output);
    lenet->H2_9.forward(&lenet->H2_9, &lenet->H1_11.output);
    lenet->H2_9.forward(&lenet->H2_9, &lenet->H1_12.output);

    lenet->H2_10.forward(&lenet->H2_10, &lenet->H1_2.output);
    lenet->H2_10.forward(&lenet->H2_10, &lenet->H1_3.output);
    lenet->H2_10.forward(&lenet->H2_10, &lenet->H1_5.output);
    lenet->H2_10.forward(&lenet->H2_10, &lenet->H1_6.output);
    lenet->H2_10.forward(&lenet->H2_10, &lenet->H1_8.output);
    lenet->H2_10.forward(&lenet->H2_10, &lenet->H1_9.output);
    lenet->H2_10.forward(&lenet->H2_10, &lenet->H1_11.output);
    lenet->H2_10.forward(&lenet->H2_10, &lenet->H1_12.output);

    lenet->H2_11.forward(&lenet->H2_11, &lenet->H1_2.output);
    lenet->H2_11.forward(&lenet->H2_11, &lenet->H1_3.output);
    lenet->H2_11.forward(&lenet->H2_11, &lenet->H1_5.output);
    lenet->H2_11.forward(&lenet->H2_11, &lenet->H1_6.output);
    lenet->H2_11.forward(&lenet->H2_11, &lenet->H1_8.output);
    lenet->H2_11.forward(&lenet->H2_11, &lenet->H1_9.output);
    lenet->H2_11.forward(&lenet->H2_11, &lenet->H1_11.output);
    lenet->H2_11.forward(&lenet->H2_11, &lenet->H1_12.output);

    lenet->H2_12.forward(&lenet->H2_12, &lenet->H1_2.output);
    lenet->H2_12.forward(&lenet->H2_12, &lenet->H1_3.output);
    lenet->H2_12.forward(&lenet->H2_12, &lenet->H1_5.output);
    lenet->H2_12.forward(&lenet->H2_12, &lenet->H1_6.output);
    lenet->H2_12.forward(&lenet->H2_12, &lenet->H1_8.output);
    lenet->H2_12.forward(&lenet->H2_12, &lenet->H1_9.output);
    lenet->H2_12.forward(&lenet->H2_12, &lenet->H1_11.output);
    lenet->H2_12.forward(&lenet->H2_12, &lenet->H1_12.output);
}

LeNet LeNetInit() {
    LeNet lenet;

    lenet.forward = LeNetForward;

    lenet.H1_1 = Conv2DInit(2, true, 5, 5);
    lenet.H1_2 = Conv2DInit(2, true, 5, 5);
    lenet.H1_3 = Conv2DInit(2, true, 5, 5);
    lenet.H1_4 = Conv2DInit(2, true, 5, 5);
    lenet.H1_5 = Conv2DInit(2, true, 5, 5);
    lenet.H1_6 = Conv2DInit(2, true, 5, 5);
    lenet.H1_7 = Conv2DInit(2, true, 5, 5);
    lenet.H1_8 = Conv2DInit(2, true, 5, 5);
    lenet.H1_9 = Conv2DInit(2, true, 5, 5);
    lenet.H1_10 = Conv2DInit(2, true, 5, 5);
    lenet.H1_11 = Conv2DInit(2, true, 5, 5);
    lenet.H1_12 = Conv2DInit(2, true, 5, 5);

    lenet.H2_1 = Conv2DInit(2, true, 5, 5);
    lenet.H2_2 = Conv2DInit(2, true, 5, 5);
    lenet.H2_3 = Conv2DInit(2, true, 5, 5);
    lenet.H2_4 = Conv2DInit(2, true, 5, 5);
    lenet.H2_5 = Conv2DInit(2, true, 5, 5);
    lenet.H2_6 = Conv2DInit(2, true, 5, 5);
    lenet.H2_7 = Conv2DInit(2, true, 5, 5);
    lenet.H2_8 = Conv2DInit(2, true, 5, 5);
    lenet.H2_9 = Conv2DInit(2, true, 5, 5);
    lenet.H2_10 = Conv2DInit(2, true, 5, 5);
    lenet.H2_11 = Conv2DInit(2, true, 5, 5);
    lenet.H2_12 = Conv2DInit(2, true, 5, 5);
    return lenet;
}

int main(int argc, char *argv[]) {
    // you can obtain MNIST here http://yann.lecun.com/exdb/mnist/
    int train_samples = 7291;
    Images images = normalize(resize_bilinear(
            mnist_load_images("/Users/jan/Downloads/train-images-idx3-ubyte", 2051),
            16, 16));
    Labels labels = mnist_load_labels("/Users/jan/Downloads/train-labels-idx1-ubyte", 2049);
    if (images.count != labels.count) error_out("Number of images and labels not equal!");
    Indices indices = split_dataset(images.count, train_samples, 2007);
    LeNet lenet = LeNetInit();
    Data input, output;
    input.dims = (int[2]){images.height, images.width};
    for (int epoch = 0; epoch < 23; epoch++) {
        printf("Epoch: ");
        printf("%d\n", epoch);
        for (int img = 0; img < train_samples; img++) {
            if (img % 100 == 0) {
                double completion_rate = (double)img / train_samples;
                printf("\r");
                for (int i = 0; i < 81 * completion_rate; i++) printf("#");
            }
            input.data = images.data_float + indices.train_set[img] * images.size;
            lenet.forward(&lenet, &input, &output);
            usleep(1000);
        }
        printf("\n");
    }
    return 0;
}
