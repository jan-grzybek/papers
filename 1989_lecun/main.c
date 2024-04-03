#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <stdbool.h>

const bool do_gradient_check = false;
const float learning_rate = (float)3e-2, eps = (float)1e-4;
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
    printf("%s\n", error_log);
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
    int *dims, dims_count, items;
    float *data;
} Data;

Data copy_data(Data *data) {
    Data copy;
    copy.dims_count = data->dims_count;
    copy.items = data->items;
    copy.dims = malloc(data->dims_count * sizeof(int));
    copy.data = malloc(data->items * sizeof(float));
    for (int i = 0; i < data->dims_count; i++) copy.dims[i] = data->dims[i];
    for (int i = 0; i < data->items; i++) copy.data[i] = data->data[i];
    return copy;
}

Data copy_data_container(Data *data) {
    Data copy;
    copy.dims_count = data->dims_count;
    copy.items = data->items;
    copy.dims = malloc(data->dims_count * sizeof(int));
    copy.data = malloc(data->items * sizeof(float));
    for (int i = 0; i < data->dims_count; i++) copy.dims[i] = data->dims[i];
    return copy;
}

void data_deallocate(Data *data) {
    if (data->dims != NULL) free(data->dims);
    data->dims = NULL;
    if (data->data != NULL) free(data->data);
    data->data = NULL;
}

typedef struct Conv2D {
    int stride, height, width, weights_size, bias_size;
    bool padding;
    Data output;
    float *weights, *bias, *weights_grad, *bias_grad;;
    void (*kernel_forward)(struct Conv2D*, Data*);
    Data (*kernel_backward)(struct Conv2D*, Data*, Data*, bool);
    void (*bias_activation_forward)(struct Conv2D*);
    Data (*bias_activation_backward)(struct Conv2D*, Data*);
    void (*reset_output)(struct Conv2D*, Data*);
    void (*update)(struct Conv2D*);
} Conv2D;

void fill_array_random_floats(float *array, int n, double range_start, double range_end) {
    if (range_end < range_start) error_out("range_end smaller than range_start");
    double range = range_end - range_start;
    for (int idx = 0; idx < n; idx++) array[idx] = (float)(range * rand() / RAND_MAX + range_start);
}

void fill_array_zeros(float *array, int n) {
    for (int idx = 0; idx < n; idx++) array[idx] = 0;
}

void Conv2DResetOutput(Conv2D *layer, Data *input) {
    if (layer->output.data != NULL) data_deallocate(&layer->output);
    int dims_count = 2;
    layer->output.dims = malloc(dims_count * sizeof(int));
    layer->output.dims_count = dims_count;
    if (layer->padding) {
        int out_h = (input->dims[0] - 1) / layer->stride + 1;
        int out_w = (input->dims[1] - 1) / layer->stride + 1;
        layer->output.dims[0] = out_h;
        layer->output.dims[1] = out_w;
    } else {
        if (input->dims[0] < layer->height) error_out("input dim [0] smaller than kernel height");
        if (input->dims[1] < layer->width) error_out("input dim [1] smaller than kernel width");
        int out_h = (input->dims[0] - layer->height) / layer->stride + 1;
        int out_w = (input->dims[1] - layer->width) / layer->stride + 1;
        layer->output.dims[0] = out_h;
        layer->output.dims[1] = out_w;
    }
    int output_size = layer->output.dims[0] * layer->output.dims[1];
    layer->output.items = output_size;
    layer->output.data = malloc(output_size * sizeof(float));
    fill_array_zeros(layer->output.data, output_size);
    layer->bias_size = output_size;
    if (layer->bias == NULL) {
        layer->bias = malloc(output_size * sizeof(float));
        fill_array_zeros(layer->bias, output_size);
    }
    if (layer->bias_grad == NULL) {
        layer->bias_grad = malloc(output_size * sizeof(float));
        fill_array_zeros(layer->bias_grad, output_size);
    }
}

void Conv2DKernelForward(Conv2D *layer, Data *input) {
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

Data Conv2DKernelBackward(Conv2D *layer, Data *input, Data *upper_derivative, bool input_layer) {
    Data lower_derivative;
    if (!input_layer) {
        lower_derivative = copy_data_container(input);
        fill_array_zeros(lower_derivative.data, input->items);
    }
    int x_start = (layer->padding) ? -layer->width / 2 : 0;
    int y_start = (layer->padding) ? -layer->height / 2 : 0;
    int out_idx = 0;
    for (int y = y_start; y < y_start + layer->output.dims[0] * layer->stride; y += layer->stride) {
        for (int x = x_start; x < x_start + layer->output.dims[1] * layer->stride; x += layer->stride) {
            for (int yk = 0; yk < layer->height; yk++) {
                for (int xk = 0; xk < layer->width; xk++) {
                    int x_shifted = x + xk;
                    int y_shifted = y + yk;
                    if (x_shifted < 0 || y_shifted < 0 || input->dims[1] <= x_shifted || input->dims[0] <= y_shifted)
                        layer->weights_grad[yk * layer->width + xk] -= upper_derivative->data[out_idx];
                    else {
                        layer->weights_grad[yk * layer->width + xk] +=
                                input->data[y_shifted * input->dims[1] + x_shifted] * upper_derivative->data[out_idx];
                        if (!input_layer) lower_derivative.data[y_shifted * input->dims[1] + x_shifted] +=
                                layer->weights[yk * layer->width + xk] * upper_derivative->data[out_idx];
                    }
                }
            }
            out_idx++;
        }
    }
    return lower_derivative;
}

void Conv2DBiasActivationForward(Conv2D *layer) {
    int output_size = layer->output.dims[0] * layer->output.dims[1];
    for (int idx = 0; idx < output_size; idx++) {
        layer->output.data[idx] += layer->bias[idx];
        layer->output.data[idx] = (float)tanh((double)layer->output.data[idx]);
    }
}

Data Conv2DBiasActivationBackward(Conv2D *layer, Data *upper_derivative) {
    Data lower_derivative;
    lower_derivative = copy_data_container(&layer->output);
    for (int idx = 0; idx < layer->output.items; idx++) {
        float tanh_derivative = (1 - layer->output.data[idx] * layer->output.data[idx]) * upper_derivative->data[idx];
        layer->bias_grad[idx] += tanh_derivative;
        lower_derivative.data[idx] = tanh_derivative;
    }
    return lower_derivative;
}

void Conv2DUpdate(Conv2D *layer) {
    for (int idx = 0; idx < layer->weights_size; idx++) layer->weights[idx] -= learning_rate * layer->weights_grad[idx];
    for (int idx = 0; idx < layer->bias_size; idx++) layer->bias[idx] -= learning_rate * layer->bias_grad[idx];
    fill_array_zeros(layer->weights_grad, layer->weights_size);
    fill_array_zeros(layer->bias_grad, layer->bias_size);
}

Conv2D Conv2DInit(int stride, bool padding, int height, int width, int total_kernels) {
    if (stride < 1) error_out("Conv2D stride has to be >= 1");
    if (height < 1) error_out("kernel's height has to be >= 1");
    if (width < 1) error_out("kernel's width has to be >= 1");
    Conv2D conv;
    conv.kernel_forward = Conv2DKernelForward;
    conv.kernel_backward = Conv2DKernelBackward;
    conv.bias_activation_forward = Conv2DBiasActivationForward;
    conv.bias_activation_backward = Conv2DBiasActivationBackward;
    conv.reset_output = Conv2DResetOutput;
    conv.update = Conv2DUpdate;
    conv.stride = stride;
    conv.padding = padding;
    conv.height = height;
    conv.width = width;
    int kernel_size = height * width;
    conv.weights_size = kernel_size;
    conv.weights = malloc(kernel_size * sizeof(float));
    fill_array_random_floats(conv.weights, kernel_size,
                             -2.4/(double)(kernel_size * total_kernels),
                             2.4/(double)(kernel_size * total_kernels));
    conv.weights_grad = malloc(kernel_size * sizeof(float));
    fill_array_zeros(conv.weights_grad, kernel_size);
    conv.bias = NULL;
    conv.bias_grad = NULL;
    conv.output.data = NULL;
    // for (int i =0; i < 25; i++) conv.weights[i] = i % 2;
    return conv;
}

typedef struct FC {
    Data output;
    int forward_offset, backward_offset, weights_size, bias_size;
    float *weights, *bias, *weights_grad, *bias_grad;
    void (*madd_forward)(struct FC*, Data*);
    Data (*madd_backward)(struct FC*, Data*, Data*);
    void (*bias_activation_forward)(struct FC*);
    Data (*bias_activation_backward)(struct FC*, Data*);
    void (*reset_output)(struct FC*);
    void (*update)(struct FC*);
} FC;

void FCResetOutput(FC *layer) {
    if (layer->output.data != NULL) {
        free(layer->output.data);
        layer->output.data = NULL;
    }
    layer->output.items = layer->output.dims[0];
    layer->output.data = malloc(layer->output.dims[0] * sizeof(float));
    fill_array_zeros(layer->output.data, layer->output.dims[0]);
    layer->forward_offset = 0;
    layer->backward_offset = 0;
}

void FCUpdate(FC *layer) {
    for (int idx = 0; idx < layer->weights_size; idx++) layer->weights[idx] -= learning_rate * layer->weights_grad[idx];
    for (int idx = 0; idx < layer->bias_size; idx++) layer->bias[idx] -= learning_rate * layer->bias_grad[idx];
    fill_array_zeros(layer->weights_grad, layer->weights_size);
    fill_array_zeros(layer->bias_grad, layer->bias_size);
}

void FCMAddForward(FC *layer, Data *input) {
    for (int out_idx = 0; out_idx < layer->output.dims[0]; out_idx++) {
        int out_offset = out_idx * input->items + layer->forward_offset;
        for (int in_idx = 0; in_idx < input->items; in_idx++)
            layer->output.data[out_idx] += layer->weights[out_offset + in_idx] * input->data[in_idx];
    }
    layer->forward_offset += layer->output.dims[0] * input->items;
}

Data FCMAddBackward(FC *layer, Data *input, Data *upper_derivative) {
    Data lower_derivative;
    lower_derivative = copy_data_container(input);
    fill_array_zeros(lower_derivative.data, input->items);
    for (int out_idx = 0; out_idx < layer->output.dims[0]; out_idx++) {
        int out_offset = out_idx * input->items + layer->backward_offset;
        for (int in_idx = 0; in_idx < input->items; in_idx++) {
            layer->weights_grad[out_offset + in_idx] += input->data[in_idx] * upper_derivative->data[out_idx];
            lower_derivative.data[in_idx] += layer->weights[out_offset + in_idx] * upper_derivative->data[out_idx];
        }
    }
    layer->backward_offset += layer->output.dims[0] * input->items;
    return lower_derivative;
}

void FCBiasActivationForward(FC *layer) {
    for (int idx = 0; idx < layer->output.dims[0]; idx++) {
        layer->output.data[idx] += layer->bias[idx];
        layer->output.data[idx] = (float)tanh((double)layer->output.data[idx]);
    }
}

Data FCBiasActivationBackward(FC *layer, Data *upper_derivative) {
    Data lower_derivative;
    lower_derivative = copy_data_container(&layer->output);
    for (int idx = 0; idx < layer->output.dims[0]; idx++) {
        float tanh_derivative = (1 - layer->output.data[idx] * layer->output.data[idx]) * upper_derivative->data[idx];
        layer->bias_grad[idx] += tanh_derivative;
        lower_derivative.data[idx] = tanh_derivative;
    }
    return lower_derivative;
}

FC FCInit(int input_units, int output_units) {
    if (input_units < 1) error_out("FC must have >= 1 input units");
    if (output_units < 1) error_out("FC must have >= 1 output units");
    FC fc;
    fc.madd_forward = FCMAddForward;
    fc.madd_backward = FCMAddBackward;
    fc.bias_activation_forward = FCBiasActivationForward;
    fc.bias_activation_backward = FCBiasActivationBackward;
    fc.reset_output = FCResetOutput;
    fc.update = FCUpdate;

    int fc_size = input_units * output_units;
    fc.weights_size = fc_size;
    fc.weights = malloc(fc_size * sizeof(float));
    fill_array_random_floats(fc.weights, fc_size,
                             -2.4/(double)(input_units),
                             2.4/(double)(input_units));
    fc.weights_grad = malloc(fc_size * sizeof(float));
    fill_array_zeros(fc.weights_grad, fc_size);
    fc.bias_size = output_units;
    fc.bias = malloc(output_units * sizeof(float));
    fill_array_zeros(fc.bias, output_units);
    fc.bias_grad = malloc(output_units * sizeof(float));
    fill_array_zeros(fc.bias_grad, output_units);
    fc.output.dims = malloc(sizeof(int));
    fc.output.dims[0] = output_units;
    fc.output.dims_count = 1;
    fc.output.data = NULL;
    // for (int i =0; i < 25; i++) conv.weights[i] = i % 2;
    return fc;
}

float MSEForward(Data *observed, Data *target) {
    if (observed->dims_count != target->dims_count)
        error_out("observed and target arrays in MSE are not of equal dimensions");
    for (int dim_idx = 0; dim_idx < observed->dims_count; dim_idx++) {
        if (observed->dims[dim_idx] != target->dims[dim_idx])
            error_out("observed and target arrays in MSE are not of equal dimensions");
    }
    float total_diff = 0;
    for (int idx = 0; idx < observed->items; idx++) {
        float diff = observed->data[idx] - target->data[idx];
        total_diff += diff * diff;
    }
    return total_diff / (float)observed->items;
}

Data MSEBackward(Data *observed, Data *target) {
    Data derivative;
    derivative = copy_data_container(observed);
    for (int idx = 0; idx < observed->items; idx++)
        derivative.data[idx] =  2 * (observed->data[idx] - target->data[idx]) / (float)observed->items;
    return derivative;
}

typedef struct LeNet {
    Conv2D H1_1, H1_2, H1_3, H1_4, H1_5, H1_6, H1_7, H1_8, H1_9, H1_10, H1_11, H1_12;
    Conv2D H2_1, H2_2, H2_3, H2_4, H2_5, H2_6, H2_7, H2_8, H2_9, H2_10, H2_11, H2_12;
    FC FC1, FC2;
    Data (*forward)(struct LeNet*, Data*);
    void (*backward)(struct LeNet*, Data*, Data);
    void (*update)(struct LeNet*);
    void (*grad_check)(struct LeNet*, Data*, Data*);
} LeNet;

Data LeNetForward(LeNet *lenet, Data *input) {
    // A long time ago in a galaxy far, far away ...
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

    lenet->H1_1.kernel_forward(&lenet->H1_1, input);
    lenet->H1_2.kernel_forward(&lenet->H1_2, input);
    lenet->H1_3.kernel_forward(&lenet->H1_3, input);
    lenet->H1_4.kernel_forward(&lenet->H1_4, input);
    lenet->H1_5.kernel_forward(&lenet->H1_5, input);
    lenet->H1_6.kernel_forward(&lenet->H1_6, input);
    lenet->H1_7.kernel_forward(&lenet->H1_7, input);
    lenet->H1_8.kernel_forward(&lenet->H1_8, input);
    lenet->H1_9.kernel_forward(&lenet->H1_9, input);
    lenet->H1_10.kernel_forward(&lenet->H1_10, input);
    lenet->H1_11.kernel_forward(&lenet->H1_11, input);
    lenet->H1_12.kernel_forward(&lenet->H1_12, input);

    lenet->H1_1.bias_activation_forward(&lenet->H1_1);
    lenet->H1_2.bias_activation_forward(&lenet->H1_2);
    lenet->H1_3.bias_activation_forward(&lenet->H1_3);
    lenet->H1_4.bias_activation_forward(&lenet->H1_4);
    lenet->H1_5.bias_activation_forward(&lenet->H1_5);
    lenet->H1_6.bias_activation_forward(&lenet->H1_6);
    lenet->H1_7.bias_activation_forward(&lenet->H1_7);
    lenet->H1_8.bias_activation_forward(&lenet->H1_8);
    lenet->H1_9.bias_activation_forward(&lenet->H1_9);
    lenet->H1_10.bias_activation_forward(&lenet->H1_10);
    lenet->H1_11.bias_activation_forward(&lenet->H1_11);
    lenet->H1_12.bias_activation_forward(&lenet->H1_12);

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

    lenet->H2_1.kernel_forward(&lenet->H2_1, &lenet->H1_1.output);
    lenet->H2_1.kernel_forward(&lenet->H2_1, &lenet->H1_2.output);
    lenet->H2_1.kernel_forward(&lenet->H2_1, &lenet->H1_4.output);
    lenet->H2_1.kernel_forward(&lenet->H2_1, &lenet->H1_5.output);
    lenet->H2_1.kernel_forward(&lenet->H2_1, &lenet->H1_7.output);
    lenet->H2_1.kernel_forward(&lenet->H2_1, &lenet->H1_8.output);
    lenet->H2_1.kernel_forward(&lenet->H2_1, &lenet->H1_10.output);
    lenet->H2_1.kernel_forward(&lenet->H2_1, &lenet->H1_11.output);

    lenet->H2_2.kernel_forward(&lenet->H2_2, &lenet->H1_1.output);
    lenet->H2_2.kernel_forward(&lenet->H2_2, &lenet->H1_2.output);
    lenet->H2_2.kernel_forward(&lenet->H2_2, &lenet->H1_4.output);
    lenet->H2_2.kernel_forward(&lenet->H2_2, &lenet->H1_5.output);
    lenet->H2_2.kernel_forward(&lenet->H2_2, &lenet->H1_7.output);
    lenet->H2_2.kernel_forward(&lenet->H2_2, &lenet->H1_8.output);
    lenet->H2_2.kernel_forward(&lenet->H2_2, &lenet->H1_10.output);
    lenet->H2_2.kernel_forward(&lenet->H2_2, &lenet->H1_11.output);

    lenet->H2_3.kernel_forward(&lenet->H2_3, &lenet->H1_1.output);
    lenet->H2_3.kernel_forward(&lenet->H2_3, &lenet->H1_2.output);
    lenet->H2_3.kernel_forward(&lenet->H2_3, &lenet->H1_4.output);
    lenet->H2_3.kernel_forward(&lenet->H2_3, &lenet->H1_5.output);
    lenet->H2_3.kernel_forward(&lenet->H2_3, &lenet->H1_7.output);
    lenet->H2_3.kernel_forward(&lenet->H2_3, &lenet->H1_8.output);
    lenet->H2_3.kernel_forward(&lenet->H2_3, &lenet->H1_10.output);
    lenet->H2_3.kernel_forward(&lenet->H2_3, &lenet->H1_11.output);

    lenet->H2_4.kernel_forward(&lenet->H2_4, &lenet->H1_1.output);
    lenet->H2_4.kernel_forward(&lenet->H2_4, &lenet->H1_2.output);
    lenet->H2_4.kernel_forward(&lenet->H2_4, &lenet->H1_4.output);
    lenet->H2_4.kernel_forward(&lenet->H2_4, &lenet->H1_5.output);
    lenet->H2_4.kernel_forward(&lenet->H2_4, &lenet->H1_7.output);
    lenet->H2_4.kernel_forward(&lenet->H2_4, &lenet->H1_8.output);
    lenet->H2_4.kernel_forward(&lenet->H2_4, &lenet->H1_10.output);
    lenet->H2_4.kernel_forward(&lenet->H2_4, &lenet->H1_11.output);

    lenet->H2_5.kernel_forward(&lenet->H2_5, &lenet->H1_1.output);
    lenet->H2_5.kernel_forward(&lenet->H2_5, &lenet->H1_3.output);
    lenet->H2_5.kernel_forward(&lenet->H2_5, &lenet->H1_4.output);
    lenet->H2_5.kernel_forward(&lenet->H2_5, &lenet->H1_6.output);
    lenet->H2_5.kernel_forward(&lenet->H2_5, &lenet->H1_7.output);
    lenet->H2_5.kernel_forward(&lenet->H2_5, &lenet->H1_9.output);
    lenet->H2_5.kernel_forward(&lenet->H2_5, &lenet->H1_10.output);
    lenet->H2_5.kernel_forward(&lenet->H2_5, &lenet->H1_12.output);

    lenet->H2_6.kernel_forward(&lenet->H2_6, &lenet->H1_1.output);
    lenet->H2_6.kernel_forward(&lenet->H2_6, &lenet->H1_3.output);
    lenet->H2_6.kernel_forward(&lenet->H2_6, &lenet->H1_4.output);
    lenet->H2_6.kernel_forward(&lenet->H2_6, &lenet->H1_6.output);
    lenet->H2_6.kernel_forward(&lenet->H2_6, &lenet->H1_7.output);
    lenet->H2_6.kernel_forward(&lenet->H2_6, &lenet->H1_9.output);
    lenet->H2_6.kernel_forward(&lenet->H2_6, &lenet->H1_10.output);
    lenet->H2_6.kernel_forward(&lenet->H2_6, &lenet->H1_12.output);

    lenet->H2_7.kernel_forward(&lenet->H2_7, &lenet->H1_1.output);
    lenet->H2_7.kernel_forward(&lenet->H2_7, &lenet->H1_3.output);
    lenet->H2_7.kernel_forward(&lenet->H2_7, &lenet->H1_4.output);
    lenet->H2_7.kernel_forward(&lenet->H2_7, &lenet->H1_6.output);
    lenet->H2_7.kernel_forward(&lenet->H2_7, &lenet->H1_7.output);
    lenet->H2_7.kernel_forward(&lenet->H2_7, &lenet->H1_9.output);
    lenet->H2_7.kernel_forward(&lenet->H2_7, &lenet->H1_10.output);
    lenet->H2_7.kernel_forward(&lenet->H2_7, &lenet->H1_12.output);

    lenet->H2_8.kernel_forward(&lenet->H2_8, &lenet->H1_1.output);
    lenet->H2_8.kernel_forward(&lenet->H2_8, &lenet->H1_3.output);
    lenet->H2_8.kernel_forward(&lenet->H2_8, &lenet->H1_4.output);
    lenet->H2_8.kernel_forward(&lenet->H2_8, &lenet->H1_6.output);
    lenet->H2_8.kernel_forward(&lenet->H2_8, &lenet->H1_7.output);
    lenet->H2_8.kernel_forward(&lenet->H2_8, &lenet->H1_9.output);
    lenet->H2_8.kernel_forward(&lenet->H2_8, &lenet->H1_10.output);
    lenet->H2_8.kernel_forward(&lenet->H2_8, &lenet->H1_12.output);

    lenet->H2_9.kernel_forward(&lenet->H2_9, &lenet->H1_2.output);
    lenet->H2_9.kernel_forward(&lenet->H2_9, &lenet->H1_3.output);
    lenet->H2_9.kernel_forward(&lenet->H2_9, &lenet->H1_5.output);
    lenet->H2_9.kernel_forward(&lenet->H2_9, &lenet->H1_6.output);
    lenet->H2_9.kernel_forward(&lenet->H2_9, &lenet->H1_8.output);
    lenet->H2_9.kernel_forward(&lenet->H2_9, &lenet->H1_9.output);
    lenet->H2_9.kernel_forward(&lenet->H2_9, &lenet->H1_11.output);
    lenet->H2_9.kernel_forward(&lenet->H2_9, &lenet->H1_12.output);

    lenet->H2_10.kernel_forward(&lenet->H2_10, &lenet->H1_2.output);
    lenet->H2_10.kernel_forward(&lenet->H2_10, &lenet->H1_3.output);
    lenet->H2_10.kernel_forward(&lenet->H2_10, &lenet->H1_5.output);
    lenet->H2_10.kernel_forward(&lenet->H2_10, &lenet->H1_6.output);
    lenet->H2_10.kernel_forward(&lenet->H2_10, &lenet->H1_8.output);
    lenet->H2_10.kernel_forward(&lenet->H2_10, &lenet->H1_9.output);
    lenet->H2_10.kernel_forward(&lenet->H2_10, &lenet->H1_11.output);
    lenet->H2_10.kernel_forward(&lenet->H2_10, &lenet->H1_12.output);

    lenet->H2_11.kernel_forward(&lenet->H2_11, &lenet->H1_2.output);
    lenet->H2_11.kernel_forward(&lenet->H2_11, &lenet->H1_3.output);
    lenet->H2_11.kernel_forward(&lenet->H2_11, &lenet->H1_5.output);
    lenet->H2_11.kernel_forward(&lenet->H2_11, &lenet->H1_6.output);
    lenet->H2_11.kernel_forward(&lenet->H2_11, &lenet->H1_8.output);
    lenet->H2_11.kernel_forward(&lenet->H2_11, &lenet->H1_9.output);
    lenet->H2_11.kernel_forward(&lenet->H2_11, &lenet->H1_11.output);
    lenet->H2_11.kernel_forward(&lenet->H2_11, &lenet->H1_12.output);

    lenet->H2_12.kernel_forward(&lenet->H2_12, &lenet->H1_2.output);
    lenet->H2_12.kernel_forward(&lenet->H2_12, &lenet->H1_3.output);
    lenet->H2_12.kernel_forward(&lenet->H2_12, &lenet->H1_5.output);
    lenet->H2_12.kernel_forward(&lenet->H2_12, &lenet->H1_6.output);
    lenet->H2_12.kernel_forward(&lenet->H2_12, &lenet->H1_8.output);
    lenet->H2_12.kernel_forward(&lenet->H2_12, &lenet->H1_9.output);
    lenet->H2_12.kernel_forward(&lenet->H2_12, &lenet->H1_11.output);
    lenet->H2_12.kernel_forward(&lenet->H2_12, &lenet->H1_12.output);

    lenet->H2_1.bias_activation_forward(&lenet->H2_1);
    lenet->H2_2.bias_activation_forward(&lenet->H2_2);
    lenet->H2_3.bias_activation_forward(&lenet->H2_3);
    lenet->H2_4.bias_activation_forward(&lenet->H2_4);
    lenet->H2_5.bias_activation_forward(&lenet->H2_5);
    lenet->H2_6.bias_activation_forward(&lenet->H2_6);
    lenet->H2_7.bias_activation_forward(&lenet->H2_7);
    lenet->H2_8.bias_activation_forward(&lenet->H2_8);
    lenet->H2_9.bias_activation_forward(&lenet->H2_9);
    lenet->H2_10.bias_activation_forward(&lenet->H2_10);
    lenet->H2_11.bias_activation_forward(&lenet->H2_11);
    lenet->H2_12.bias_activation_forward(&lenet->H2_12);

    lenet->FC1.reset_output(&lenet->FC1);
    lenet->FC1.madd_forward(&lenet->FC1, &lenet->H2_1.output);
    lenet->FC1.madd_forward(&lenet->FC1, &lenet->H2_2.output);
    lenet->FC1.madd_forward(&lenet->FC1, &lenet->H2_3.output);
    lenet->FC1.madd_forward(&lenet->FC1, &lenet->H2_4.output);
    lenet->FC1.madd_forward(&lenet->FC1, &lenet->H2_5.output);
    lenet->FC1.madd_forward(&lenet->FC1, &lenet->H2_6.output);
    lenet->FC1.madd_forward(&lenet->FC1, &lenet->H2_7.output);
    lenet->FC1.madd_forward(&lenet->FC1, &lenet->H2_8.output);
    lenet->FC1.madd_forward(&lenet->FC1, &lenet->H2_9.output);
    lenet->FC1.madd_forward(&lenet->FC1, &lenet->H2_10.output);
    lenet->FC1.madd_forward(&lenet->FC1, &lenet->H2_11.output);
    lenet->FC1.madd_forward(&lenet->FC1, &lenet->H2_12.output);
    lenet->FC1.bias_activation_forward(&lenet->FC1);

    lenet->FC2.reset_output(&lenet->FC2);
    lenet->FC2.madd_forward(&lenet->FC2, &lenet->FC1.output);
    lenet->FC2.bias_activation_forward(&lenet->FC2);

    return copy_data(&lenet->FC2.output);
}

void LeNetUpdate(LeNet *lenet) {
    lenet->FC2.update(&lenet->FC2);
    lenet->FC1.update(&lenet->FC1);
    lenet->H2_12.update(&lenet->H2_12);
    lenet->H2_11.update(&lenet->H2_11);
    lenet->H2_10.update(&lenet->H2_10);
    lenet->H2_9.update(&lenet->H2_9);
    lenet->H2_8.update(&lenet->H2_8);
    lenet->H2_7.update(&lenet->H2_7);
    lenet->H2_6.update(&lenet->H2_6);
    lenet->H2_5.update(&lenet->H2_5);
    lenet->H2_4.update(&lenet->H2_4);
    lenet->H2_3.update(&lenet->H2_3);
    lenet->H2_2.update(&lenet->H2_2);
    lenet->H2_1.update(&lenet->H2_1);
    lenet->H1_12.update(&lenet->H1_12);
    lenet->H1_11.update(&lenet->H1_11);
    lenet->H1_10.update(&lenet->H1_10);
    lenet->H1_9.update(&lenet->H1_9);
    lenet->H1_8.update(&lenet->H1_8);
    lenet->H1_7.update(&lenet->H1_7);
    lenet->H1_6.update(&lenet->H1_6);
    lenet->H1_5.update(&lenet->H1_5);
    lenet->H1_4.update(&lenet->H1_4);
    lenet->H1_3.update(&lenet->H1_3);
    lenet->H1_2.update(&lenet->H1_2);
    lenet->H1_1.update(&lenet->H1_1);
}

void LeNetBackward(LeNet *lenet, Data *input, Data loss_derivative) {
    Data d0 = lenet->FC2.bias_activation_backward(&lenet->FC2, &loss_derivative);
    data_deallocate(&loss_derivative);
    Data d1 = lenet->FC2.madd_backward(&lenet->FC2, &lenet->FC1.output, &d0);
    data_deallocate(&d0);

    d0 = lenet->FC1.bias_activation_backward(&lenet->FC1, &d1);
    data_deallocate(&d1);
    d1 = lenet->FC1.madd_backward(&lenet->FC1, &lenet->H2_1.output, &d0);
    Data d2 = lenet->FC1.madd_backward(&lenet->FC1, &lenet->H2_2.output, &d0);
    Data d3 = lenet->FC1.madd_backward(&lenet->FC1, &lenet->H2_3.output, &d0);
    Data d4 = lenet->FC1.madd_backward(&lenet->FC1, &lenet->H2_4.output, &d0);
    Data d5 = lenet->FC1.madd_backward(&lenet->FC1, &lenet->H2_5.output, &d0);
    Data d6 = lenet->FC1.madd_backward(&lenet->FC1, &lenet->H2_6.output, &d0);
    Data d7 = lenet->FC1.madd_backward(&lenet->FC1, &lenet->H2_7.output, &d0);
    Data d8 = lenet->FC1.madd_backward(&lenet->FC1, &lenet->H2_8.output, &d0);
    Data d9 = lenet->FC1.madd_backward(&lenet->FC1, &lenet->H2_9.output, &d0);
    Data d10 = lenet->FC1.madd_backward(&lenet->FC1, &lenet->H2_10.output, &d0);
    Data d11 = lenet->FC1.madd_backward(&lenet->FC1, &lenet->H2_11.output, &d0);
    Data d12 = lenet->FC1.madd_backward(&lenet->FC1, &lenet->H2_12.output, &d0);
    data_deallocate(&d0);

    d0 = lenet->H2_1.bias_activation_backward(&lenet->H2_1, &d1);
    data_deallocate(&d1);
    d1 = lenet->H2_1.kernel_backward(&lenet->H2_1, &lenet->H1_1.output, &d0, false);
    Data d13 = lenet->H2_1.kernel_backward(&lenet->H2_1, &lenet->H1_2.output, &d0, false);
    Data d14 = lenet->H2_1.kernel_backward(&lenet->H2_1, &lenet->H1_4.output, &d0, false);
    Data d15 = lenet->H2_1.kernel_backward(&lenet->H2_1, &lenet->H1_5.output, &d0, false);
    Data d16 = lenet->H2_1.kernel_backward(&lenet->H2_1, &lenet->H1_7.output, &d0, false);
    Data d17 = lenet->H2_1.kernel_backward(&lenet->H2_1, &lenet->H1_8.output, &d0, false);
    Data d18 = lenet->H2_1.kernel_backward(&lenet->H2_1, &lenet->H1_10.output, &d0, false);
    Data d19 = lenet->H2_1.kernel_backward(&lenet->H2_1, &lenet->H1_11.output, &d0, false);
    data_deallocate(&d0);

    d0 = lenet->H1_1.bias_activation_backward(&lenet->H1_1, &d1);
    data_deallocate(&d1);
    lenet->H1_1.kernel_backward(&lenet->H1_1, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_2.bias_activation_backward(&lenet->H1_2, &d13);
    data_deallocate(&d13);
    lenet->H1_2.kernel_backward(&lenet->H1_2, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_4.bias_activation_backward(&lenet->H1_4, &d14);
    data_deallocate(&d14);
    lenet->H1_4.kernel_backward(&lenet->H1_4, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_5.bias_activation_backward(&lenet->H1_5, &d15);
    data_deallocate(&d15);
    lenet->H1_5.kernel_backward(&lenet->H1_5, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_7.bias_activation_backward(&lenet->H1_7, &d16);
    data_deallocate(&d16);
    lenet->H1_7.kernel_backward(&lenet->H1_7, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_8.bias_activation_backward(&lenet->H1_8, &d17);
    data_deallocate(&d17);
    lenet->H1_8.kernel_backward(&lenet->H1_8, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_10.bias_activation_backward(&lenet->H1_10, &d18);
    data_deallocate(&d18);
    lenet->H1_10.kernel_backward(&lenet->H1_10, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_11.bias_activation_backward(&lenet->H1_11, &d19);
    data_deallocate(&d19);
    lenet->H1_11.kernel_backward(&lenet->H1_11, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H2_2.bias_activation_backward(&lenet->H2_2, &d2);
    data_deallocate(&d2);
    d1 = lenet->H2_2.kernel_backward(&lenet->H2_2, &lenet->H1_1.output, &d0, false);
    d2 = lenet->H2_2.kernel_backward(&lenet->H2_2, &lenet->H1_2.output, &d0, false);
    d13 = lenet->H2_2.kernel_backward(&lenet->H2_2, &lenet->H1_4.output, &d0, false);
    d14 = lenet->H2_2.kernel_backward(&lenet->H2_2, &lenet->H1_5.output, &d0, false);
    d15 = lenet->H2_2.kernel_backward(&lenet->H2_2, &lenet->H1_7.output, &d0, false);
    d16 = lenet->H2_2.kernel_backward(&lenet->H2_2, &lenet->H1_8.output, &d0, false);
    d17 = lenet->H2_2.kernel_backward(&lenet->H2_2, &lenet->H1_10.output, &d0, false);
    d18 = lenet->H2_2.kernel_backward(&lenet->H2_2, &lenet->H1_11.output, &d0, false);
    data_deallocate(&d0);

    d0 = lenet->H1_1.bias_activation_backward(&lenet->H1_1, &d1);
    data_deallocate(&d1);
    lenet->H1_1.kernel_backward(&lenet->H1_1, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_2.bias_activation_backward(&lenet->H1_2, &d2);
    data_deallocate(&d2);
    lenet->H1_2.kernel_backward(&lenet->H1_2, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_4.bias_activation_backward(&lenet->H1_4, &d13);
    data_deallocate(&d13);
    lenet->H1_4.kernel_backward(&lenet->H1_4, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_5.bias_activation_backward(&lenet->H1_5, &d14);
    data_deallocate(&d14);
    lenet->H1_5.kernel_backward(&lenet->H1_5, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_7.bias_activation_backward(&lenet->H1_7, &d15);
    data_deallocate(&d15);
    lenet->H1_7.kernel_backward(&lenet->H1_7, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_8.bias_activation_backward(&lenet->H1_8, &d16);
    data_deallocate(&d16);
    lenet->H1_8.kernel_backward(&lenet->H1_8, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_10.bias_activation_backward(&lenet->H1_10, &d17);
    data_deallocate(&d17);
    lenet->H1_10.kernel_backward(&lenet->H1_10, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_11.bias_activation_backward(&lenet->H1_11, &d18);
    data_deallocate(&d18);
    lenet->H1_11.kernel_backward(&lenet->H1_11, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H2_3.bias_activation_backward(&lenet->H2_3, &d3);
    data_deallocate(&d3);
    d1 = lenet->H2_3.kernel_backward(&lenet->H2_3, &lenet->H1_1.output, &d0, false);
    d2 = lenet->H2_3.kernel_backward(&lenet->H2_3, &lenet->H1_2.output, &d0, false);
    d3 = lenet->H2_3.kernel_backward(&lenet->H2_3, &lenet->H1_4.output, &d0, false);
    d13 = lenet->H2_3.kernel_backward(&lenet->H2_3, &lenet->H1_5.output, &d0, false);
    d14 = lenet->H2_3.kernel_backward(&lenet->H2_3, &lenet->H1_7.output, &d0, false);
    d15 = lenet->H2_3.kernel_backward(&lenet->H2_3, &lenet->H1_8.output, &d0, false);
    d16 = lenet->H2_3.kernel_backward(&lenet->H2_3, &lenet->H1_10.output, &d0, false);
    d17 = lenet->H2_3.kernel_backward(&lenet->H2_3, &lenet->H1_11.output, &d0, false);
    data_deallocate(&d0);

    d0 = lenet->H1_1.bias_activation_backward(&lenet->H1_1, &d1);
    data_deallocate(&d1);
    lenet->H1_1.kernel_backward(&lenet->H1_1, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_2.bias_activation_backward(&lenet->H1_2, &d2);
    data_deallocate(&d2);
    lenet->H1_2.kernel_backward(&lenet->H1_2, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_4.bias_activation_backward(&lenet->H1_4, &d3);
    data_deallocate(&d3);
    lenet->H1_4.kernel_backward(&lenet->H1_4, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_5.bias_activation_backward(&lenet->H1_5, &d13);
    data_deallocate(&d13);
    lenet->H1_5.kernel_backward(&lenet->H1_5, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_7.bias_activation_backward(&lenet->H1_7, &d14);
    data_deallocate(&d14);
    lenet->H1_7.kernel_backward(&lenet->H1_7, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_8.bias_activation_backward(&lenet->H1_8, &d15);
    data_deallocate(&d15);
    lenet->H1_8.kernel_backward(&lenet->H1_8, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_10.bias_activation_backward(&lenet->H1_10, &d16);
    data_deallocate(&d16);
    lenet->H1_10.kernel_backward(&lenet->H1_10, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_11.bias_activation_backward(&lenet->H1_11, &d17);
    data_deallocate(&d17);
    lenet->H1_11.kernel_backward(&lenet->H1_11, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H2_4.bias_activation_backward(&lenet->H2_4, &d4);
    data_deallocate(&d4);
    d1 = lenet->H2_4.kernel_backward(&lenet->H2_4, &lenet->H1_1.output, &d0, false);
    d2 = lenet->H2_4.kernel_backward(&lenet->H2_4, &lenet->H1_2.output, &d0, false);
    d3 = lenet->H2_4.kernel_backward(&lenet->H2_4, &lenet->H1_4.output, &d0, false);
    d4 = lenet->H2_4.kernel_backward(&lenet->H2_4, &lenet->H1_5.output, &d0, false);
    d13 = lenet->H2_4.kernel_backward(&lenet->H2_4, &lenet->H1_7.output, &d0, false);
    d14 = lenet->H2_4.kernel_backward(&lenet->H2_4, &lenet->H1_8.output, &d0, false);
    d15 = lenet->H2_4.kernel_backward(&lenet->H2_4, &lenet->H1_10.output, &d0, false);
    d16 = lenet->H2_4.kernel_backward(&lenet->H2_4, &lenet->H1_11.output, &d0, false);
    data_deallocate(&d0);

    d0 = lenet->H1_1.bias_activation_backward(&lenet->H1_1, &d1);
    data_deallocate(&d1);
    lenet->H1_1.kernel_backward(&lenet->H1_1, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_2.bias_activation_backward(&lenet->H1_2, &d2);
    data_deallocate(&d2);
    lenet->H1_2.kernel_backward(&lenet->H1_2, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_4.bias_activation_backward(&lenet->H1_4, &d3);
    data_deallocate(&d3);
    lenet->H1_4.kernel_backward(&lenet->H1_4, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_5.bias_activation_backward(&lenet->H1_5, &d4);
    data_deallocate(&d4);
    lenet->H1_5.kernel_backward(&lenet->H1_5, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_7.bias_activation_backward(&lenet->H1_7, &d13);
    data_deallocate(&d13);
    lenet->H1_7.kernel_backward(&lenet->H1_7, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_8.bias_activation_backward(&lenet->H1_8, &d14);
    data_deallocate(&d14);
    lenet->H1_8.kernel_backward(&lenet->H1_8, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_10.bias_activation_backward(&lenet->H1_10, &d15);
    data_deallocate(&d15);
    lenet->H1_10.kernel_backward(&lenet->H1_10, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_11.bias_activation_backward(&lenet->H1_11, &d16);
    data_deallocate(&d16);
    lenet->H1_11.kernel_backward(&lenet->H1_11, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H2_5.bias_activation_backward(&lenet->H2_5, &d5);
    data_deallocate(&d5);
    d1 = lenet->H2_5.kernel_backward(&lenet->H2_5, &lenet->H1_1.output, &d0, false);
    d2 = lenet->H2_5.kernel_backward(&lenet->H2_5, &lenet->H1_3.output, &d0, false);
    d3 = lenet->H2_5.kernel_backward(&lenet->H2_5, &lenet->H1_4.output, &d0, false);
    d4 = lenet->H2_5.kernel_backward(&lenet->H2_5, &lenet->H1_6.output, &d0, false);
    d5 = lenet->H2_5.kernel_backward(&lenet->H2_5, &lenet->H1_7.output, &d0, false);
    d13 = lenet->H2_5.kernel_backward(&lenet->H2_5, &lenet->H1_9.output, &d0, false);
    d14 = lenet->H2_5.kernel_backward(&lenet->H2_5, &lenet->H1_10.output, &d0, false);
    d15 = lenet->H2_5.kernel_backward(&lenet->H2_5, &lenet->H1_12.output, &d0, false);
    data_deallocate(&d0);

    d0 = lenet->H1_1.bias_activation_backward(&lenet->H1_1, &d1);
    data_deallocate(&d1);
    lenet->H1_1.kernel_backward(&lenet->H1_1, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_3.bias_activation_backward(&lenet->H1_3, &d2);
    data_deallocate(&d2);
    lenet->H1_3.kernel_backward(&lenet->H1_3, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_4.bias_activation_backward(&lenet->H1_4, &d3);
    data_deallocate(&d3);
    lenet->H1_4.kernel_backward(&lenet->H1_4, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_6.bias_activation_backward(&lenet->H1_6, &d4);
    data_deallocate(&d4);
    lenet->H1_6.kernel_backward(&lenet->H1_6, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_7.bias_activation_backward(&lenet->H1_7, &d5);
    data_deallocate(&d5);
    lenet->H1_7.kernel_backward(&lenet->H1_7, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_9.bias_activation_backward(&lenet->H1_9, &d13);
    data_deallocate(&d13);
    lenet->H1_9.kernel_backward(&lenet->H1_9, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_10.bias_activation_backward(&lenet->H1_10, &d14);
    data_deallocate(&d14);
    lenet->H1_10.kernel_backward(&lenet->H1_10, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_12.bias_activation_backward(&lenet->H1_12, &d15);
    data_deallocate(&d15);
    lenet->H1_12.kernel_backward(&lenet->H1_12, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H2_6.bias_activation_backward(&lenet->H2_6, &d6);
    data_deallocate(&d6);
    d1 = lenet->H2_6.kernel_backward(&lenet->H2_6, &lenet->H1_1.output, &d0, false);
    d2 = lenet->H2_6.kernel_backward(&lenet->H2_6, &lenet->H1_3.output, &d0, false);
    d3 = lenet->H2_6.kernel_backward(&lenet->H2_6, &lenet->H1_4.output, &d0, false);
    d4 = lenet->H2_6.kernel_backward(&lenet->H2_6, &lenet->H1_6.output, &d0, false);
    d5 = lenet->H2_6.kernel_backward(&lenet->H2_6, &lenet->H1_7.output, &d0, false);
    d6 = lenet->H2_6.kernel_backward(&lenet->H2_6, &lenet->H1_9.output, &d0, false);
    d13 = lenet->H2_6.kernel_backward(&lenet->H2_6, &lenet->H1_10.output, &d0, false);
    d14 = lenet->H2_6.kernel_backward(&lenet->H2_6, &lenet->H1_12.output, &d0, false);
    data_deallocate(&d0);

    d0 = lenet->H1_1.bias_activation_backward(&lenet->H1_1, &d1);
    data_deallocate(&d1);
    lenet->H1_1.kernel_backward(&lenet->H1_1, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_3.bias_activation_backward(&lenet->H1_3, &d2);
    data_deallocate(&d2);
    lenet->H1_3.kernel_backward(&lenet->H1_3, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_4.bias_activation_backward(&lenet->H1_4, &d3);
    data_deallocate(&d3);
    lenet->H1_4.kernel_backward(&lenet->H1_4, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_6.bias_activation_backward(&lenet->H1_6, &d4);
    data_deallocate(&d4);
    lenet->H1_6.kernel_backward(&lenet->H1_6, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_7.bias_activation_backward(&lenet->H1_7, &d5);
    data_deallocate(&d5);
    lenet->H1_7.kernel_backward(&lenet->H1_7, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_9.bias_activation_backward(&lenet->H1_9, &d6);
    data_deallocate(&d6);
    lenet->H1_9.kernel_backward(&lenet->H1_9, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_10.bias_activation_backward(&lenet->H1_10, &d13);
    data_deallocate(&d13);
    lenet->H1_10.kernel_backward(&lenet->H1_10, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_12.bias_activation_backward(&lenet->H1_12, &d14);
    data_deallocate(&d14);
    lenet->H1_12.kernel_backward(&lenet->H1_12, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H2_7.bias_activation_backward(&lenet->H2_7, &d7);
    data_deallocate(&d7);
    d1 = lenet->H2_7.kernel_backward(&lenet->H2_7, &lenet->H1_1.output, &d0, false);
    d2 = lenet->H2_7.kernel_backward(&lenet->H2_7, &lenet->H1_3.output, &d0, false);
    d3 = lenet->H2_7.kernel_backward(&lenet->H2_7, &lenet->H1_4.output, &d0, false);
    d4 = lenet->H2_7.kernel_backward(&lenet->H2_7, &lenet->H1_6.output, &d0, false);
    d5 = lenet->H2_7.kernel_backward(&lenet->H2_7, &lenet->H1_7.output, &d0, false);
    d6 = lenet->H2_7.kernel_backward(&lenet->H2_7, &lenet->H1_9.output, &d0, false);
    d7 = lenet->H2_7.kernel_backward(&lenet->H2_7, &lenet->H1_10.output, &d0, false);
    d13 = lenet->H2_7.kernel_backward(&lenet->H2_7, &lenet->H1_12.output, &d0, false);
    data_deallocate(&d0);

    d0 = lenet->H1_1.bias_activation_backward(&lenet->H1_1, &d1);
    data_deallocate(&d1);
    lenet->H1_1.kernel_backward(&lenet->H1_1, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_3.bias_activation_backward(&lenet->H1_3, &d2);
    data_deallocate(&d2);
    lenet->H1_3.kernel_backward(&lenet->H1_3, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_4.bias_activation_backward(&lenet->H1_4, &d3);
    data_deallocate(&d3);
    lenet->H1_4.kernel_backward(&lenet->H1_4, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_6.bias_activation_backward(&lenet->H1_6, &d4);
    data_deallocate(&d4);
    lenet->H1_6.kernel_backward(&lenet->H1_6, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_7.bias_activation_backward(&lenet->H1_7, &d5);
    data_deallocate(&d5);
    lenet->H1_7.kernel_backward(&lenet->H1_7, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_9.bias_activation_backward(&lenet->H1_9, &d6);
    data_deallocate(&d6);
    lenet->H1_9.kernel_backward(&lenet->H1_9, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_10.bias_activation_backward(&lenet->H1_10, &d7);
    data_deallocate(&d7);
    lenet->H1_10.kernel_backward(&lenet->H1_10, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_12.bias_activation_backward(&lenet->H1_12, &d13);
    data_deallocate(&d13);
    lenet->H1_12.kernel_backward(&lenet->H1_12, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H2_8.bias_activation_backward(&lenet->H2_8, &d8);
    data_deallocate(&d8);
    d1 = lenet->H2_8.kernel_backward(&lenet->H2_8, &lenet->H1_1.output, &d0, false);
    d2 = lenet->H2_8.kernel_backward(&lenet->H2_8, &lenet->H1_3.output, &d0, false);
    d3 = lenet->H2_8.kernel_backward(&lenet->H2_8, &lenet->H1_4.output, &d0, false);
    d4 = lenet->H2_8.kernel_backward(&lenet->H2_8, &lenet->H1_6.output, &d0, false);
    d5 = lenet->H2_8.kernel_backward(&lenet->H2_8, &lenet->H1_7.output, &d0, false);
    d6 = lenet->H2_8.kernel_backward(&lenet->H2_8, &lenet->H1_9.output, &d0, false);
    d7 = lenet->H2_8.kernel_backward(&lenet->H2_8, &lenet->H1_10.output, &d0, false);
    d8 = lenet->H2_8.kernel_backward(&lenet->H2_8, &lenet->H1_12.output, &d0, false);
    data_deallocate(&d0);

    d0 = lenet->H1_1.bias_activation_backward(&lenet->H1_1, &d1);
    data_deallocate(&d1);
    lenet->H1_1.kernel_backward(&lenet->H1_1, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_3.bias_activation_backward(&lenet->H1_3, &d2);
    data_deallocate(&d2);
    lenet->H1_3.kernel_backward(&lenet->H1_3, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_4.bias_activation_backward(&lenet->H1_4, &d3);
    data_deallocate(&d3);
    lenet->H1_4.kernel_backward(&lenet->H1_4, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_6.bias_activation_backward(&lenet->H1_6, &d4);
    data_deallocate(&d4);
    lenet->H1_6.kernel_backward(&lenet->H1_6, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_7.bias_activation_backward(&lenet->H1_7, &d5);
    data_deallocate(&d5);
    lenet->H1_7.kernel_backward(&lenet->H1_7, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_9.bias_activation_backward(&lenet->H1_9, &d6);
    data_deallocate(&d6);
    lenet->H1_9.kernel_backward(&lenet->H1_9, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_10.bias_activation_backward(&lenet->H1_10, &d7);
    data_deallocate(&d7);
    lenet->H1_10.kernel_backward(&lenet->H1_10, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_12.bias_activation_backward(&lenet->H1_12, &d8);
    data_deallocate(&d8);
    lenet->H1_12.kernel_backward(&lenet->H1_12, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H2_9.bias_activation_backward(&lenet->H2_9, &d9);
    data_deallocate(&d9);
    d1 = lenet->H2_9.kernel_backward(&lenet->H2_9, &lenet->H1_2.output, &d0, false);
    d2 = lenet->H2_9.kernel_backward(&lenet->H2_9, &lenet->H1_3.output, &d0, false);
    d3 = lenet->H2_9.kernel_backward(&lenet->H2_9, &lenet->H1_5.output, &d0, false);
    d4 = lenet->H2_9.kernel_backward(&lenet->H2_9, &lenet->H1_6.output, &d0, false);
    d5 = lenet->H2_9.kernel_backward(&lenet->H2_9, &lenet->H1_8.output, &d0, false);
    d6 = lenet->H2_9.kernel_backward(&lenet->H2_9, &lenet->H1_9.output, &d0, false);
    d7 = lenet->H2_9.kernel_backward(&lenet->H2_9, &lenet->H1_11.output, &d0, false);
    d8 = lenet->H2_9.kernel_backward(&lenet->H2_9, &lenet->H1_12.output, &d0, false);
    data_deallocate(&d0);

    d0 = lenet->H1_2.bias_activation_backward(&lenet->H1_2, &d1);
    data_deallocate(&d1);
    lenet->H1_2.kernel_backward(&lenet->H1_2, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_3.bias_activation_backward(&lenet->H1_3, &d2);
    data_deallocate(&d2);
    lenet->H1_3.kernel_backward(&lenet->H1_3, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_5.bias_activation_backward(&lenet->H1_5, &d3);
    data_deallocate(&d3);
    lenet->H1_5.kernel_backward(&lenet->H1_5, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_6.bias_activation_backward(&lenet->H1_6, &d4);
    data_deallocate(&d4);
    lenet->H1_6.kernel_backward(&lenet->H1_6, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_8.bias_activation_backward(&lenet->H1_8, &d5);
    data_deallocate(&d5);
    lenet->H1_8.kernel_backward(&lenet->H1_8, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_9.bias_activation_backward(&lenet->H1_9, &d6);
    data_deallocate(&d6);
    lenet->H1_9.kernel_backward(&lenet->H1_9, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_11.bias_activation_backward(&lenet->H1_11, &d7);
    data_deallocate(&d7);
    lenet->H1_11.kernel_backward(&lenet->H1_11, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_12.bias_activation_backward(&lenet->H1_12, &d8);
    data_deallocate(&d8);
    lenet->H1_12.kernel_backward(&lenet->H1_12, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H2_10.bias_activation_backward(&lenet->H2_10, &d10);
    data_deallocate(&d10);
    d1 = lenet->H2_10.kernel_backward(&lenet->H2_10, &lenet->H1_2.output, &d0, false);
    d2 = lenet->H2_10.kernel_backward(&lenet->H2_10, &lenet->H1_3.output, &d0, false);
    d3 = lenet->H2_10.kernel_backward(&lenet->H2_10, &lenet->H1_5.output, &d0, false);
    d4 = lenet->H2_10.kernel_backward(&lenet->H2_10, &lenet->H1_6.output, &d0, false);
    d5 = lenet->H2_10.kernel_backward(&lenet->H2_10, &lenet->H1_8.output, &d0, false);
    d6 = lenet->H2_10.kernel_backward(&lenet->H2_10, &lenet->H1_9.output, &d0, false);
    d7 = lenet->H2_10.kernel_backward(&lenet->H2_10, &lenet->H1_11.output, &d0, false);
    d8 = lenet->H2_10.kernel_backward(&lenet->H2_10, &lenet->H1_12.output, &d0, false);
    data_deallocate(&d0);

    d0 = lenet->H1_2.bias_activation_backward(&lenet->H1_2, &d1);
    data_deallocate(&d1);
    lenet->H1_2.kernel_backward(&lenet->H1_2, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_3.bias_activation_backward(&lenet->H1_3, &d2);
    data_deallocate(&d2);
    lenet->H1_3.kernel_backward(&lenet->H1_3, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_5.bias_activation_backward(&lenet->H1_5, &d3);
    data_deallocate(&d3);
    lenet->H1_5.kernel_backward(&lenet->H1_5, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_6.bias_activation_backward(&lenet->H1_6, &d4);
    data_deallocate(&d4);
    lenet->H1_6.kernel_backward(&lenet->H1_6, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_8.bias_activation_backward(&lenet->H1_8, &d5);
    data_deallocate(&d5);
    lenet->H1_8.kernel_backward(&lenet->H1_8, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_9.bias_activation_backward(&lenet->H1_9, &d6);
    data_deallocate(&d6);
    lenet->H1_9.kernel_backward(&lenet->H1_9, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_11.bias_activation_backward(&lenet->H1_11, &d7);
    data_deallocate(&d7);
    lenet->H1_11.kernel_backward(&lenet->H1_11, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_12.bias_activation_backward(&lenet->H1_12, &d8);
    data_deallocate(&d8);
    lenet->H1_12.kernel_backward(&lenet->H1_12, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H2_11.bias_activation_backward(&lenet->H2_11, &d11);
    data_deallocate(&d11);
    d1 = lenet->H2_11.kernel_backward(&lenet->H2_11, &lenet->H1_2.output, &d0, false);
    d2 = lenet->H2_11.kernel_backward(&lenet->H2_11, &lenet->H1_3.output, &d0, false);
    d3 = lenet->H2_11.kernel_backward(&lenet->H2_11, &lenet->H1_5.output, &d0, false);
    d4 = lenet->H2_11.kernel_backward(&lenet->H2_11, &lenet->H1_6.output, &d0, false);
    d5 = lenet->H2_11.kernel_backward(&lenet->H2_11, &lenet->H1_8.output, &d0, false);
    d6 = lenet->H2_11.kernel_backward(&lenet->H2_11, &lenet->H1_9.output, &d0, false);
    d7 = lenet->H2_11.kernel_backward(&lenet->H2_11, &lenet->H1_11.output, &d0, false);
    d8 = lenet->H2_11.kernel_backward(&lenet->H2_11, &lenet->H1_12.output, &d0, false);
    data_deallocate(&d0);

    d0 = lenet->H1_2.bias_activation_backward(&lenet->H1_2, &d1);
    data_deallocate(&d1);
    lenet->H1_2.kernel_backward(&lenet->H1_2, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_3.bias_activation_backward(&lenet->H1_3, &d2);
    data_deallocate(&d2);
    lenet->H1_3.kernel_backward(&lenet->H1_3, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_5.bias_activation_backward(&lenet->H1_5, &d3);
    data_deallocate(&d3);
    lenet->H1_5.kernel_backward(&lenet->H1_5, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_6.bias_activation_backward(&lenet->H1_6, &d4);
    data_deallocate(&d4);
    lenet->H1_6.kernel_backward(&lenet->H1_6, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_8.bias_activation_backward(&lenet->H1_8, &d5);
    data_deallocate(&d5);
    lenet->H1_8.kernel_backward(&lenet->H1_8, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_9.bias_activation_backward(&lenet->H1_9, &d6);
    data_deallocate(&d6);
    lenet->H1_9.kernel_backward(&lenet->H1_9, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_11.bias_activation_backward(&lenet->H1_11, &d7);
    data_deallocate(&d7);
    lenet->H1_11.kernel_backward(&lenet->H1_11, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_12.bias_activation_backward(&lenet->H1_12, &d8);
    data_deallocate(&d8);
    lenet->H1_12.kernel_backward(&lenet->H1_12, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H2_12.bias_activation_backward(&lenet->H2_12, &d12);
    data_deallocate(&d12);
    d1 = lenet->H2_12.kernel_backward(&lenet->H2_12, &lenet->H1_2.output, &d0, false);
    d2 = lenet->H2_12.kernel_backward(&lenet->H2_12, &lenet->H1_3.output, &d0, false);
    d3 = lenet->H2_12.kernel_backward(&lenet->H2_12, &lenet->H1_5.output, &d0, false);
    d4 = lenet->H2_12.kernel_backward(&lenet->H2_12, &lenet->H1_6.output, &d0, false);
    d5 = lenet->H2_12.kernel_backward(&lenet->H2_12, &lenet->H1_8.output, &d0, false);
    d6 = lenet->H2_12.kernel_backward(&lenet->H2_12, &lenet->H1_9.output, &d0, false);
    d7 = lenet->H2_12.kernel_backward(&lenet->H2_12, &lenet->H1_11.output, &d0, false);
    d8 = lenet->H2_12.kernel_backward(&lenet->H2_12, &lenet->H1_12.output, &d0, false);
    data_deallocate(&d0);

    d0 = lenet->H1_2.bias_activation_backward(&lenet->H1_2, &d1);
    data_deallocate(&d1);
    lenet->H1_2.kernel_backward(&lenet->H1_2, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_3.bias_activation_backward(&lenet->H1_3, &d2);
    data_deallocate(&d2);
    lenet->H1_3.kernel_backward(&lenet->H1_3, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_5.bias_activation_backward(&lenet->H1_5, &d3);
    data_deallocate(&d3);
    lenet->H1_5.kernel_backward(&lenet->H1_5, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_6.bias_activation_backward(&lenet->H1_6, &d4);
    data_deallocate(&d4);
    lenet->H1_6.kernel_backward(&lenet->H1_6, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_8.bias_activation_backward(&lenet->H1_8, &d5);
    data_deallocate(&d5);
    lenet->H1_8.kernel_backward(&lenet->H1_8, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_9.bias_activation_backward(&lenet->H1_9, &d6);
    data_deallocate(&d6);
    lenet->H1_9.kernel_backward(&lenet->H1_9, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_11.bias_activation_backward(&lenet->H1_11, &d7);
    data_deallocate(&d7);
    lenet->H1_11.kernel_backward(&lenet->H1_11, input, &d0, true);
    data_deallocate(&d0);

    d0 = lenet->H1_12.bias_activation_backward(&lenet->H1_12, &d8);
    data_deallocate(&d8);
    lenet->H1_12.kernel_backward(&lenet->H1_12, input, &d0, true);
    data_deallocate(&d0);
}

void check_grad(LeNet *lenet, Data *input, Data *target, float *parameters, float *grads, int size) {
    for (int i = 0; i < size; i++) {
        // change var +eps
        float tmp = parameters[i];
        parameters[i] += eps;
        Data output = lenet->forward(lenet, input);
        float loss_0 = MSEForward(&output, target);
        data_deallocate(&output);
        // change var -2*eps
        parameters[i] = tmp - eps;
        output = lenet->forward(lenet, input);
        float loss_1 = MSEForward(&output, target);
        data_deallocate(&output);
        // reset parameter val
        parameters[i] = tmp;
        double est_grad = (loss_0 - loss_1) / (2 * eps);
        double magnitude = fmax(fabs(est_grad), fabs((double)grads[i]));
        double rel_diff = fabs(est_grad - grads[i]) / magnitude;
        if (rel_diff > 1e-1) {
            printf("idx [%d], rel diff = %f\n", i, rel_diff);
            printf("%f\n", est_grad);
            printf("%f\n", grads[i]);
        }
    }
}

void LeNetGradCheck(LeNet *lenet, Data *input, Data *target) {
    printf("\nGrad check: FC2.bias\n");
    check_grad(lenet, input, target, lenet->FC2.bias, lenet->FC2.bias_grad, lenet->FC2.bias_size);
    printf("\nGrad check: FC2.weights\n");
    check_grad(lenet, input, target, lenet->FC2.weights, lenet->FC2.weights_grad, lenet->FC2.weights_size);
    printf("\nGrad check: FC1.bias\n");
    check_grad(lenet, input, target, lenet->FC1.bias, lenet->FC1.bias_grad, lenet->FC1.bias_size);
    printf("\nGrad check: FC1.weights\n");
    check_grad(lenet, input, target, lenet->FC1.weights, lenet->FC1.weights_grad, lenet->FC1.weights_size);
    printf("\nGrad check: H2_1.bias\n");
    check_grad(lenet, input, target, lenet->H2_1.bias, lenet->H2_1.bias_grad, lenet->H2_1.bias_size);
    printf("\nGrad check: H2_1.weights\n");
    check_grad(lenet, input, target, lenet->H2_1.weights, lenet->H2_1.weights_grad, lenet->H2_1.weights_size);
    printf("\nGrad check: H2_12.bias\n");
    check_grad(lenet, input, target, lenet->H2_12.bias, lenet->H2_12.bias_grad, lenet->H2_12.bias_size);
    printf("\nGrad check: H2_12.weights\n");
    check_grad(lenet, input, target, lenet->H2_12.weights, lenet->H2_12.weights_grad, lenet->H2_12.weights_size);
    printf("\nGrad check: H1_12.bias\n");
    check_grad(lenet, input, target, lenet->H1_12.bias, lenet->H1_12.bias_grad, lenet->H1_12.bias_size);
    printf("\nGrad check: H1_12.weights\n");
    check_grad(lenet, input, target, lenet->H1_12.weights, lenet->H1_12.weights_grad, lenet->H1_12.weights_size);
    exit(0);
}

LeNet LeNetInit() {
    LeNet lenet;

    lenet.forward = LeNetForward;
    lenet.backward = LeNetBackward;
    lenet.update = LeNetUpdate;
    lenet.grad_check = LeNetGradCheck;

    lenet.H1_1 = Conv2DInit(2, true, 5, 5, 1);
    lenet.H1_2 = Conv2DInit(2, true, 5, 5, 1);
    lenet.H1_3 = Conv2DInit(2, true, 5, 5, 1);
    lenet.H1_4 = Conv2DInit(2, true, 5, 5, 1);
    lenet.H1_5 = Conv2DInit(2, true, 5, 5, 1);
    lenet.H1_6 = Conv2DInit(2, true, 5, 5, 1);
    lenet.H1_7 = Conv2DInit(2, true, 5, 5, 1);
    lenet.H1_8 = Conv2DInit(2, true, 5, 5, 1);
    lenet.H1_9 = Conv2DInit(2, true, 5, 5, 1);
    lenet.H1_10 = Conv2DInit(2, true, 5, 5, 1);
    lenet.H1_11 = Conv2DInit(2, true, 5, 5, 1);
    lenet.H1_12 = Conv2DInit(2, true, 5, 5, 1);

    lenet.H2_1 = Conv2DInit(2, true, 5, 5, 8);
    lenet.H2_2 = Conv2DInit(2, true, 5, 5, 8);
    lenet.H2_3 = Conv2DInit(2, true, 5, 5, 8);
    lenet.H2_4 = Conv2DInit(2, true, 5, 5, 8);
    lenet.H2_5 = Conv2DInit(2, true, 5, 5, 8);
    lenet.H2_6 = Conv2DInit(2, true, 5, 5, 8);
    lenet.H2_7 = Conv2DInit(2, true, 5, 5, 8);
    lenet.H2_8 = Conv2DInit(2, true, 5, 5, 8);
    lenet.H2_9 = Conv2DInit(2, true, 5, 5, 8);
    lenet.H2_10 = Conv2DInit(2, true, 5, 5, 8);
    lenet.H2_11 = Conv2DInit(2, true, 5, 5, 8);
    lenet.H2_12 = Conv2DInit(2, true, 5, 5, 8);

    lenet.FC1 = FCInit(12 * 5 * 5, 30);
    lenet.FC2 = FCInit(30, 10);
    return lenet;
}

Data mnist_get_target_vector(int label) {
    int classes = 10;
    Data target;
    target.dims = malloc(sizeof(int));
    target.dims[0] = classes;
    target.dims_count = 1;
    target.items = classes;
    target.data = malloc(classes * sizeof(float));
    for (int i = 0; i < classes; i++) {
        if (i == label) target.data[i] = 1;
        else target.data[i] = -1;
    }
    return target;
}

int get_prediction(Data *output) {
    int argmax = 0;
    double max_val = -2.0;
    for (int i = 0; i < output->dims[0]; i++) {
        if (output->data[i] > max_val) {
            max_val = output->data[i];
            argmax = i;
        }
    }
    return argmax;
}

void show_kernel(const float *weights, int height, int width, int index) {
    int size = width * height;
    float max_value = -1000, min_value = 1000;
    for (int i = 0; i < size; i++) {
        if (weights[i] > max_value) max_value = weights[i];
        if (weights[i] < min_value) min_value = weights[i];
    }
    float range = max_value - min_value;
    print_maybe("\nKernel %d, %d x %d\n", index, height, width);
    for (int col = 0; col < width+2; col++) print_maybe("# #"); print_maybe("\n");
    for (int row = 0; row < height; row++) {
        print_maybe("#  ");
        int row_offset = row * width;
        for (int col = 0; col < width; col++) {
            int idx = row_offset + col;
            float val = (weights[idx] - min_value) / range;
            if (val > 0.9) print_maybe("@@@");
            else if (val > 0.8) print_maybe("111");
            else if (val > 0.7) print_maybe("...");
            else if (val > 0.3) print_maybe(" . ");
            else if (val > 0.2) print_maybe("...");
            else if (val > 0.1) print_maybe("111");
            else print_maybe("@@@");
        }
        print_maybe("  #");
        print_maybe("\n");
    }
    for (int col = 0; col < width+2; col++) print_maybe("# #"); print_maybe("\n");
}

int main(int argc, char *argv[]) {
    // you can obtain MNIST here http://yann.lecun.com/exdb/mnist/
    int train_samples = 7291;
    int test_samples = 2007;
    Images images = normalize(resize_bilinear(
            mnist_load_images(argv[1], 2051),
            16, 16));
    Labels labels = mnist_load_labels(argv[2], 2049);
    if (images.count != labels.count) error_out("Number of images and labels not equal!");
    Indices indices = split_dataset(images.count, train_samples, test_samples);
    LeNet lenet = LeNetInit();
    Data input;
    input.dims = (int[2]){images.height, images.width};
    for (int epoch = 0; epoch < 23; epoch++) {
        printf("\nEpoch: ");
        printf("%d\n", epoch);
        for (int img = 0; img < train_samples; img++) {
            int index = indices.train_set[img];
            input.data = images.data_float + index * images.size;
            Data output = lenet.forward(&lenet, &input);
            Data target = mnist_get_target_vector(labels.data[index]);
            float loss = MSEForward(&output, &target);
            if (img % 1000 == 0) print_maybe("Loss [MSE]: %f\n", loss);
            lenet.backward(&lenet, &input, MSEBackward(&output, &target));
            if (do_gradient_check && loss < 0.1) lenet.grad_check(&lenet, &input, &target);
            lenet.update(&lenet);
            data_deallocate(&output);
            data_deallocate(&target);
        }

        float total_loss = 0;
        int misclassified_count = 0;
        for (int img = 0; img < train_samples; img++) {
            int index = indices.train_set[img];
            input.data = images.data_float + index * images.size;
            Data output = lenet.forward(&lenet, &input);
            Data target = mnist_get_target_vector(labels.data[index]);
            misclassified_count += get_prediction(&output) != labels.data[index];
            total_loss += MSEForward(&output, &target);
            data_deallocate(&output);
            data_deallocate(&target);
        }
        printf("\nAvg loss training set [MSE]: %f", total_loss / (float)train_samples);
        printf("\nMisclassified patterns training set: %.2f%%\n",
               (float)misclassified_count * 100 / (float)train_samples);

        total_loss = 0;
        misclassified_count = 0;
        for (int img = 0; img < test_samples; img++) {
            int index = indices.test_set[img];
            input.data = images.data_float + index * images.size;
            Data output = lenet.forward(&lenet, &input);
            Data target = mnist_get_target_vector(labels.data[index]);
            misclassified_count += get_prediction(&output) != labels.data[index];
            total_loss += MSEForward(&output, &target);
            data_deallocate(&output);
            data_deallocate(&target);
        }
        printf("\nAvg loss test set [MSE]: %f", total_loss / (float)train_samples);
        printf("\nMisclassified patterns test set: %.2f%%\n", (float)misclassified_count * 100 / (float)train_samples);

        show_kernel(lenet.H1_1.weights, lenet.H1_1.height, lenet.H1_1.width, 1);
        show_kernel(lenet.H1_2.weights, lenet.H1_2.height, lenet.H1_2.width, 2);
        show_kernel(lenet.H1_3.weights, lenet.H1_3.height, lenet.H1_3.width, 3);
        show_kernel(lenet.H1_4.weights, lenet.H1_4.height, lenet.H1_4.width, 4);
        show_kernel(lenet.H1_5.weights, lenet.H1_5.height, lenet.H1_5.width, 5);
        show_kernel(lenet.H1_6.weights, lenet.H1_6.height, lenet.H1_6.width, 6);
        show_kernel(lenet.H1_7.weights, lenet.H1_7.height, lenet.H1_7.width, 7);
        show_kernel(lenet.H1_8.weights, lenet.H1_8.height, lenet.H1_8.width, 8);
        show_kernel(lenet.H1_9.weights, lenet.H1_9.height, lenet.H1_9.width, 9);
        show_kernel(lenet.H1_10.weights, lenet.H1_10.height, lenet.H1_10.width, 10);
        show_kernel(lenet.H1_11.weights, lenet.H1_11.height, lenet.H1_11.width, 11);
        show_kernel(lenet.H1_12.weights, lenet.H1_12.height, lenet.H1_12.width, 12);
    }
    return 0;
}
