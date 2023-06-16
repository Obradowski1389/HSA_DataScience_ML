from scripts.img_to_npy import img_to_npy
from scripts.neural_network import build_nn


img_in = "img/provided/slo.png"
npy_out = "npy/out/out_slo.npy"

npy_in = "npy/input/in_ns.npy"
out_img_name = "out_ns.png"


def main():
    img_to_npy(img_in, npy_out)

    # first arg == 1 - train model; first arg != 1 - predict image
    # epochs 
    # batch_size
    build_nn(2, 150, 512, npy_in, out_img_name)


if __name__ == "__main__":
    main()
