from math import floor


def conv_size(v_in: int, kernel_size: int, stride: int, dilation: int = 1, padding: int = 0):
    numerator = v_in + 2 * padding - dilation * (kernel_size - 1) - 1
    float_out = (numerator / stride) + 1
    return floor(float_out)


if __name__ == "__main__":
    h_in = 9
    w_in = 9
    c_in = 7
    first_c_out = 32
    second_c_out = 64

    kernel_size_1 = 4
    stride_1 = 2

    first_layer_h = conv_size(h_in, kernel_size_1, stride_1)
    first_layer_w = conv_size(w_in, kernel_size_1, stride_1)
    print(f"First layer out: ({first_layer_h}, {first_layer_w}, {first_c_out})")

    kernel_size_2 = 3
    stride_2 = 1

    second_layer_h = conv_size(first_layer_h, kernel_size_2, stride_2)
    second_layer_w = conv_size(first_layer_w, kernel_size_2, stride_2)
    print(f"First layer out: ({second_layer_h}, {second_layer_w}, {second_c_out})")
