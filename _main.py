from network import Network

# http://yann.lecun.com/exdb/mnist/
def read_idx_format(images, labels):
  # TODO(rami): check the target arch's endianness
  magic_number = int.from_bytes(images.read(4), "big")
  assert magic_number == 0x803, "invalid idx3 format"
  magic_number = int.from_bytes(labels.read(4), "big")
  assert magic_number == 0x801, "invalid idx1 format"
  del magic_number  # we don't need this anymore

  result = []
  num_images = int.from_bytes(images.read(4), "big")
  num_labels = int.from_bytes(labels.read(4), "big")
  assert num_images == num_labels, "images and labels are not compatible"
  del num_labels  # we can just use one of them
  n_rows = int.from_bytes(images.read(4), "big")
  n_cols = int.from_bytes(images.read(4), "big")
  for i in range(num_images):
    image = []
    # it's stupid to specify byte order when reading a single byte
    for i in range(n_rows * n_cols):
      image.append(int.from_bytes(images.read(1), "big"))
    label = int.from_bytes(labels.read(1), "big")
    result.append((image, label))
  return result


def load_mnist():
  raw_training_images = open("data/train-images.idx3-ubyte", "rb")
  raw_training_labels = open("data/train-labels.idx1-ubyte", "rb")
  training_set = read_idx_format(raw_training_images, raw_training_labels)
  raw_training_images.close()
  raw_training_labels.close()
  raw_test_images = open("data/t10k-images.idx3-ubyte", "rb")
  raw_test_labels = open("data/t10k-labels.idx1-ubyte", "rb")
  test_set = read_idx_format(raw_test_images, raw_test_labels)
  raw_test_images.close()
  raw_test_labels.close()

  return (training_set, test_set)

def main():
  net = Network([28*28, 16, 16, 10])
  (training_set, test_set) = load_mnist()
  net.train(training_set, 1000, 0.1, 20, test_set)


if __name__ == "__main__":
  main()