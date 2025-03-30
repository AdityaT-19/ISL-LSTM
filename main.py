import torch


def main():
    print("Hello from lstm-pyt!")
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
