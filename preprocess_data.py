import csv
import glob
import os
import xml.etree.ElementTree as ET

XML_FOLDER = "annotations/xmls/"
DATASET_FOLDER = "images/"
TRAIN_OUTPUT_FILE = "train.csv"
VALIDATION_OUTPUT_FILE = "validation.csv"

SPLIT_RATIO = 0.8

def main():
  if not os.path.exists(DATASET_FOLDER):
    print("Dataset not found")
    return

  class_names = {}
  output = []
  xml_files = glob.glob("{}/*xml".format(XML_FOLDER))
  for i, xml_file in enumerate(xml_files):
    tree = ET.parse(xml_file)

    path = os.path.join(DATASET_FOLDER, tree.findtext("./filename"))

    height = int(tree.findtext("./size/height"))
    width = int(tree.findtext("./size/width"))
    xmin = int(tree.findtext("./object/bndbox/xmin"))
    ymin = int(tree.findtext("./object/bndbox/ymin"))
    xmax = int(tree.findtext("./object/bndbox/xmax"))
    ymax = int(tree.findtext("./object/bndbox/ymax"))

    basename = os.path.basename(path)
    basename = os.path.splitext(basename)[0]
    class_name = 0 if basename[0].islower() else 1

    class_names[class_name] = 'Cat' if class_name else 'Dog'

    output.append((path, height, width, xmin, ymin, xmax, ymax, class_names[class_name], class_name))

    # preserve percentage of samples for each class ("stratified")
  output.sort(key=lambda tup : tup[-1])
  lengths = []
  i = 0
  last = 0
  for j, row in enumerate(output):
    if last == row[-1]:
      i += 1
    else:
      print("class {}: {} images".format(output[j-1][-2], i))
      lengths.append(i)
      i = 1
      last += 1

  print("class {}: {} images".format(output[j-1][-2], i))
  lengths.append(i)

  with open(TRAIN_OUTPUT_FILE, "w") as train, open(VALIDATION_OUTPUT_FILE, "w") as validate:
    writer = csv.writer(train, delimiter=",")
    writer2 = csv.writer(validate, delimiter=",")

    s = 0
    for c in lengths:
      for i in range(c):
        print("{}/{}".format(s + 1, sum(lengths)), end="\r")

        path, height, width, xmin, ymin, xmax, ymax, class_id, class_name = output[s]

        if xmin >= xmax or ymin >= ymax or xmax > width or ymax > height or xmin < 0 or ymin < 0:
            print("Warning: {} contains invalid box. Skipped...".format(path))
            continue

        row = [path, height, width, xmin, ymin, xmax, ymax, class_names[class_name], class_name]
        if i <= c * SPLIT_RATIO:
          writer.writerow(row)
        else:
          writer2.writerow(row)

        s += 1

  print("\nDone!")