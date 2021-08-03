from test import *

def main():
    try:
        test_Transformer_cell()
        test_transformerencoderlayer()
        test_transformerencoderlayer_gelu()
        test_transformerencoder()
        test_transformerdecoderlayer()
        test_transformerdecoderlayer_gelu()
        test_transformerdecoder()
        print(f"All test done.")
    except RuntimeError as e:
        print(f"something goes wrong.")

if __name__ == '__main__':
    main()

