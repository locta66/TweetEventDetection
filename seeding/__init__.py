import os
import sys

seeding_package_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(seeding_package_path + '/../preprocess')
sys.path.append(seeding_package_path + '/../config')
# sys.path.append(seeding_package_path)
print('seeding imported')
