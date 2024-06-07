def array_to_cpp(array):
    return '{' + ','.join(str(x) for x in array) + '}'

def main():
    table = []
    for i in range(256):
        bit_array = [((i >> x) & 1) for x in range(8)]
        prefix_sum = [sum(bit_array[:x+1]) for x in range(8)]
        table.append(array_to_cpp(prefix_sum))
    print("const uint8_t prefix_sum_table[256][8] = {" + ','.join(table) + "};")

if __name__ == '__main__':
    main()