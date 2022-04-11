from math import floor


def conv_size(v_in: int, kernel_size: int, stride: int, dilation: int = 1, padding: int = 0):
    numerator = v_in + 2 * padding - dilation * (kernel_size - 1) - 1
    float_out = (numerator / stride) + 1
    return floor(float_out)


if __name__ == "__main__":
    # h_in = 9
    # w_in = 9
    # c_in = 7
    h_in = 17
    w_in = 17
    c_in = 6

    first_c_out = 32
    second_c_out = 64

    kernel_size_1 = 8
    stride_1 = 1

    first_layer_h = conv_size(h_in, kernel_size_1, stride_1)
    first_layer_w = conv_size(w_in, kernel_size_1, stride_1)
    print(f"First layer out: ({first_layer_h}, {first_layer_w}, {first_c_out})")

    kernel_size_2 = 6
    stride_2 = 1

    second_layer_h = conv_size(first_layer_h, kernel_size_2, stride_2)
    second_layer_w = conv_size(first_layer_w, kernel_size_2, stride_2)
    print(f"Second layer out: ({second_layer_h}, {second_layer_w}, {second_c_out})")

    kernel_size_3 = 5
    stride_3 = 1

    third_layer_h = conv_size(second_layer_h, kernel_size_3, stride_3)
    third_layer_w = conv_size(second_layer_w, kernel_size_3, stride_3)
    print(f"Third layer out: ({third_layer_h}, {third_layer_w}, {second_c_out})")
