import os
import csv
import xml.etree.ElementTree as ET

# Set the directory where the XML files are located
xml_dir = 'dataset/annotations'

# Create a list to store the data
data = []

# Loop through all the XML files in the directory
for filename in os.listdir(xml_dir):
    if filename.endswith('.xml'):
        # Parse the XML file
        tree = ET.parse(os.path.join(xml_dir, filename))
        root = tree.getroot()

        # Extract the relevant information
        image_name = root.find('filename').text
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        depth = int(root.find('size/depth').text)
        object_name = root.find('object/name').text
        xmin = float(root.find('object/bndbox/xmin').text)
        ymin = float(root.find('object/bndbox/ymin').text)
        xmax = float(root.find('object/bndbox/xmax').text)
        ymax = float(root.find('object/bndbox/ymax').text)

        # Add the data to the list
        data.append([image_name, width, height, depth, object_name, xmin, ymin, xmax, ymax])

# Write the data to a CSV file
with open('data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_name', 'width', 'height', 'depth', 'object_name', 'xmin', 'ymin', 'xmax', 'ymax'])
    writer.writerows(data)

print('CSV file created successfully!')