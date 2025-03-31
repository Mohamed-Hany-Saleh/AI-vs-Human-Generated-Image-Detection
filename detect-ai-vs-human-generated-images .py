{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "591c6d86",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2025-02-26T17:10:50.244893Z",
     "iopub.status.busy": "2025-02-26T17:10:50.244686Z",
     "iopub.status.idle": "2025-02-26T17:11:03.601544Z",
     "shell.execute_reply": "2025-02-26T17:11:03.600850Z"
    },
    "papermill": {
     "duration": 13.364195,
     "end_time": "2025-02-26T17:11:03.603088",
     "exception": false,
     "start_time": "2025-02-26T17:10:50.238893",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "np.random.seed(2)\n",
    "import random\n",
    "import io\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from keras.utils import to_categorical\n",
    "from keras.models import Sequential\n",
    "from tensorflow.keras import layers, models, regularizers\n",
    "import os\n",
    "from tensorflow.keras.preprocessing.image import load_img, img_to_array\n",
    "from keras.models import Model\n",
    "from keras.regularizers import l1_l2 \n",
    "from keras.layers import *\n",
    "import pandas as pd\n",
    "from keras.optimizers import Adam\n",
    "from keras.initializers import GlorotUniform \n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Added ReduceLROnPlateau\n",
    "from keras import models\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from PIL import Image\n",
    "from keras import layers\n",
    "from tensorflow.keras.applications import InceptionV3\n",
    "from tensorflow.keras.applications import VGG16, ResNet50, ResNet101\n",
    "from PIL import Image, ImageChops, ImageEnhance\n",
    "from tensorflow.keras.applications import EfficientNetB4\n",
    "import os\n",
    "import itertools"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "037323de",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:11:03.614074Z",
     "iopub.status.busy": "2025-02-26T17:11:03.613591Z",
     "iopub.status.idle": "2025-02-26T17:11:03.617238Z",
     "shell.execute_reply": "2025-02-26T17:11:03.616565Z"
    },
    "papermill": {
     "duration": 0.010053,
     "end_time": "2025-02-26T17:11:03.618527",
     "exception": false,
     "start_time": "2025-02-26T17:11:03.608474",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Define paths\n",
    "directory = '/kaggle/input/ai-vs-human-generated-dataset'\n",
    "train_data = '/kaggle/input/ai-vs-human-generated-dataset/train_data'\n",
    "train_csv = '/kaggle/input/ai-vs-human-generated-dataset/train.csv'\n",
    "test_data = '/kaggle/input/ai-vs-human-generated-dataset/test_data_v2'\n",
    "test_csv = '/kaggle/input/ai-vs-human-generated-dataset/test.csv'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "cd16ec5f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:11:03.628874Z",
     "iopub.status.busy": "2025-02-26T17:11:03.628634Z",
     "iopub.status.idle": "2025-02-26T17:11:03.631762Z",
     "shell.execute_reply": "2025-02-26T17:11:03.631104Z"
    },
    "papermill": {
     "duration": 0.009713,
     "end_time": "2025-02-26T17:11:03.633171",
     "exception": false,
     "start_time": "2025-02-26T17:11:03.623458",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Define image size and ELA quality\n",
    "image_size = (48, 48)  # Desired image size\n",
    "# ela_quality = 90  # ELA quality parameter"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "59c654ab",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:11:03.642892Z",
     "iopub.status.busy": "2025-02-26T17:11:03.642674Z",
     "iopub.status.idle": "2025-02-26T17:11:03.777012Z",
     "shell.execute_reply": "2025-02-26T17:11:03.776378Z"
    },
    "papermill": {
     "duration": 0.140652,
     "end_time": "2025-02-26T17:11:03.778473",
     "exception": false,
     "start_time": "2025-02-26T17:11:03.637821",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Load the training CSV file\n",
    "train_df = pd.read_csv(train_csv)\n",
    "test_df= pd.read_csv(test_csv)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "8a2b970e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:11:03.788457Z",
     "iopub.status.busy": "2025-02-26T17:11:03.788203Z",
     "iopub.status.idle": "2025-02-26T17:11:03.825459Z",
     "shell.execute_reply": "2025-02-26T17:11:03.824892Z"
    },
    "papermill": {
     "duration": 0.043359,
     "end_time": "2025-02-26T17:11:03.826629",
     "exception": false,
     "start_time": "2025-02-26T17:11:03.783270",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Remove the 'train_data/' prefix from the file_name column\n",
    "train_df['file_name'] = train_df['file_name'].str.replace('train_data/', '', regex=False)\n",
    "# Remove the 'train_data/' prefix from the file_name column\n",
    "test_df['id'] = test_df['id'].str.replace('test_data_v2/', '', regex=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "8bbc6246",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:11:03.836352Z",
     "iopub.status.busy": "2025-02-26T17:11:03.836122Z",
     "iopub.status.idle": "2025-02-26T17:11:03.839064Z",
     "shell.execute_reply": "2025-02-26T17:11:03.838464Z"
    },
    "papermill": {
     "duration": 0.008915,
     "end_time": "2025-02-26T17:11:03.840141",
     "exception": false,
     "start_time": "2025-02-26T17:11:03.831226",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "#def convert_to_ela_image(image, quality):\n",
    "#     \"\"\"Convert an image to ELA (Error Level Analysis) format.\"\"\"\n",
    "#     # Ensure the image is in uint8 format and has the correct shape\n",
    "#     if image.dtype != np.uint8:\n",
    " #        image = (image * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8\n",
    "    \n",
    "#     # Convert the NumPy array to a PIL Image\n",
    "  #   original_image = Image.fromarray(image)\n",
    "    \n",
    "#     # Save the original image to a BytesIO object with a specific quality\n",
    "   #  temp_buffer = io.BytesIO()\n",
    "    # original_image.save(temp_buffer, 'JPEG', quality=quality)\n",
    "     #temp_buffer.seek(0)\n",
    "    \n",
    "#     # Open the compressed image from the BytesIO object\n",
    "     #temp_image = Image.open(temp_buffer)\n",
    "    \n",
    "#     # Compute the ELA image\n",
    "     #ela_image = Image.fromarray(np.abs(np.array(original_image) - np.array(temp_image)))\n",
    "     #return ela_image\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "511f1fa2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:11:03.849705Z",
     "iopub.status.busy": "2025-02-26T17:11:03.849488Z",
     "iopub.status.idle": "2025-02-26T17:11:03.852855Z",
     "shell.execute_reply": "2025-02-26T17:11:03.852252Z"
    },
    "papermill": {
     "duration": 0.009482,
     "end_time": "2025-02-26T17:11:03.854061",
     "exception": false,
     "start_time": "2025-02-26T17:11:03.844579",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# # Function to preprocess an image (convert to ELA, resize, normalize)\n",
    "def ela_preprocessing(image):\n",
    "#     \"\"\"Preprocess an image by converting it to ELA, resizing, and normalizing.\"\"\"\n",
    "#     # Convert the image to ELA format\n",
    "     ela_image = convert_to_ela_image(image, ela_quality)\n",
    "    \n",
    "#     # Resize the image\n",
    "     resized_image = ela_image.resize(image_size)\n",
    "    \n",
    "#     # Normalize the image to [0, 1]\n",
    "     return np.array(resized_image) / 255.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "f62241ca",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:11:03.863552Z",
     "iopub.status.busy": "2025-02-26T17:11:03.863336Z",
     "iopub.status.idle": "2025-02-26T17:11:03.876129Z",
     "shell.execute_reply": "2025-02-26T17:11:03.875576Z"
    },
    "papermill": {
     "duration": 0.018687,
     "end_time": "2025-02-26T17:11:03.877264",
     "exception": false,
     "start_time": "2025-02-26T17:11:03.858577",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Split the dataset into two categories\n",
    "category_0 = train_df[train_df['label'] == 0]  # Images with label 0\n",
    "category_1 = train_df[train_df['label'] == 1]  # Images with label 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "16180b7e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:11:03.886727Z",
     "iopub.status.busy": "2025-02-26T17:11:03.886525Z",
     "iopub.status.idle": "2025-02-26T17:11:03.890106Z",
     "shell.execute_reply": "2025-02-26T17:11:03.889249Z"
    },
    "papermill": {
     "duration": 0.009827,
     "end_time": "2025-02-26T17:11:03.891442",
     "exception": false,
     "start_time": "2025-02-26T17:11:03.881615",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "39975\n",
      "39975\n"
     ]
    }
   ],
   "source": [
    "print(len(category_0))\n",
    "print(len(category_1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "4561c215",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:11:03.900948Z",
     "iopub.status.busy": "2025-02-26T17:11:03.900749Z",
     "iopub.status.idle": "2025-02-26T17:11:03.927492Z",
     "shell.execute_reply": "2025-02-26T17:11:03.926847Z"
    },
    "papermill": {
     "duration": 0.032957,
     "end_time": "2025-02-26T17:11:03.928843",
     "exception": false,
     "start_time": "2025-02-26T17:11:03.895886",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Randomly sample 20,000 images from each category\n",
    "num_samples = 39975\n",
    "category_0_sampled = category_0.sample(n=num_samples, random_state=42)  # Sample 20,000 from category 0\n",
    "category_1_sampled = category_1.sample(n=num_samples, random_state=42)  # Sample 20,000 from category 1\n",
    "\n",
    "# Combine the sampled data into a balanced DataFrame\n",
    "balanced_df = pd.concat([category_0_sampled, category_1_sampled])\n",
    "\n",
    "# Shuffle the balanced DataFrame\n",
    "balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "871736cc",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:11:03.938822Z",
     "iopub.status.busy": "2025-02-26T17:11:03.938614Z",
     "iopub.status.idle": "2025-02-26T17:11:03.946667Z",
     "shell.execute_reply": "2025-02-26T17:11:03.946104Z"
    },
    "papermill": {
     "duration": 0.014291,
     "end_time": "2025-02-26T17:11:03.947928",
     "exception": false,
     "start_time": "2025-02-26T17:11:03.933637",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Split the balanced dataset into training and validation sets\n",
    "train_data_df, val_data_df = train_test_split(balanced_df, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "cfa6afa2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:11:03.957624Z",
     "iopub.status.busy": "2025-02-26T17:11:03.957417Z",
     "iopub.status.idle": "2025-02-26T17:11:03.978795Z",
     "shell.execute_reply": "2025-02-26T17:11:03.978235Z"
    },
    "papermill": {
     "duration": 0.027489,
     "end_time": "2025-02-26T17:11:03.979981",
     "exception": false,
     "start_time": "2025-02-26T17:11:03.952492",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Convert the 'label' column to strings\n",
    "train_data_df['label'] = train_data_df['label'].astype(str)\n",
    "val_data_df['label'] = val_data_df['label'].astype(str)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "ae91509d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:11:03.989538Z",
     "iopub.status.busy": "2025-02-26T17:11:03.989296Z",
     "iopub.status.idle": "2025-02-26T17:11:03.993422Z",
     "shell.execute_reply": "2025-02-26T17:11:03.992837Z"
    },
    "papermill": {
     "duration": 0.010286,
     "end_time": "2025-02-26T17:11:03.994726",
     "exception": false,
     "start_time": "2025-02-26T17:11:03.984440",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Trim whitespace from filenames\n",
    "test_df['id'] = test_df['id'].str.strip()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "5ab36247",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:11:04.004279Z",
     "iopub.status.busy": "2025-02-26T17:11:04.004064Z",
     "iopub.status.idle": "2025-02-26T17:11:04.006832Z",
     "shell.execute_reply": "2025-02-26T17:11:04.006239Z"
    },
    "papermill": {
     "duration": 0.008903,
     "end_time": "2025-02-26T17:11:04.008031",
     "exception": false,
     "start_time": "2025-02-26T17:11:03.999128",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "epochs = 50\n",
    "batch_size = 128"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "6a4407c2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:11:04.018260Z",
     "iopub.status.busy": "2025-02-26T17:11:04.018018Z",
     "iopub.status.idle": "2025-02-26T17:12:45.335327Z",
     "shell.execute_reply": "2025-02-26T17:12:45.334362Z"
    },
    "papermill": {
     "duration": 101.32431,
     "end_time": "2025-02-26T17:12:45.336955",
     "exception": false,
     "start_time": "2025-02-26T17:11:04.012645",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 63960 validated image filenames belonging to 2 classes.\n",
      "Found 15990 validated image filenames belonging to 2 classes.\n",
      "Found 5540 validated image filenames.\n"
     ]
    }
   ],
   "source": [
    "# Define data augmentation for training\n",
    "train_datagen = ImageDataGenerator(\n",
    "    #preprocessing_function=ela_preprocessing,  \n",
    "    rotation_range=20,\n",
    "    width_shift_range=0.2,\n",
    "    height_shift_range=0.2,\n",
    "    shear_range=0.2,\n",
    "    zoom_range=0.2,\n",
    "    horizontal_flip=True,\n",
    "    fill_mode='nearest'\n",
    ")\n",
    "\n",
    "# Define validation and test generators (without augmentation)\n",
    "val_datagen = ImageDataGenerator(\n",
    "    #preprocessing_function=ela_preprocessing  \n",
    ")\n",
    "\n",
    "test_datagen = ImageDataGenerator(\n",
    "    #preprocessing_function=ela_preprocessing  \n",
    ")\n",
    "# Create generators for training\n",
    "train_generator = train_datagen.flow_from_dataframe(\n",
    "    dataframe=train_data_df,\n",
    "    directory=train_data,\n",
    "    x_col='file_name',\n",
    "    y_col='label',\n",
    "    target_size=image_size,  \n",
    "    batch_size=batch_size,\n",
    "    class_mode='categorical',  \n",
    "    shuffle=True  \n",
    ")\n",
    "# Create generators for validation\n",
    "val_generator = val_datagen.flow_from_dataframe(\n",
    "    dataframe=val_data_df,\n",
    "    directory=train_data,\n",
    "    x_col='file_name',\n",
    "    y_col='label',\n",
    "    target_size=image_size, \n",
    "    batch_size=batch_size,\n",
    "    class_mode='categorical',  \n",
    "    shuffle=False  \n",
    ")\n",
    "# Create the test generator again after checks\n",
    "test_generator = test_datagen.flow_from_dataframe(\n",
    "    dataframe=test_df,\n",
    "    directory=test_data,\n",
    "    x_col='id',  \n",
    "    y_col=None,  \n",
    "    target_size=image_size, \n",
    "    batch_size=batch_size,\n",
    "    class_mode=None, \n",
    "    shuffle=False \n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "78ca1ac9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:12:45.347547Z",
     "iopub.status.busy": "2025-02-26T17:12:45.347248Z",
     "iopub.status.idle": "2025-02-26T17:12:45.352287Z",
     "shell.execute_reply": "2025-02-26T17:12:45.351659Z"
    },
    "papermill": {
     "duration": 0.01145,
     "end_time": "2025-02-26T17:12:45.353413",
     "exception": false,
     "start_time": "2025-02-26T17:12:45.341963",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training generator: <keras.src.legacy.preprocessing.image.DataFrameIterator object at 0x7a75e1ad17b0>\n",
      "Validation generator: <keras.src.legacy.preprocessing.image.DataFrameIterator object at 0x7a75e1ad3f10>\n",
      "Test generator: <keras.src.legacy.preprocessing.image.DataFrameIterator object at 0x7a7653d94760>\n"
     ]
    }
   ],
   "source": [
    "# Print generator details\n",
    "print(\"Training generator:\", train_generator)\n",
    "print(\"Validation generator:\", val_generator)\n",
    "print(\"Test generator:\", test_generator)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "5284ed73",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:12:45.363657Z",
     "iopub.status.busy": "2025-02-26T17:12:45.363425Z",
     "iopub.status.idle": "2025-02-26T17:12:45.366376Z",
     "shell.execute_reply": "2025-02-26T17:12:45.365757Z"
    },
    "papermill": {
     "duration": 0.009396,
     "end_time": "2025-02-26T17:12:45.367665",
     "exception": false,
     "start_time": "2025-02-26T17:12:45.358269",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    " #def build_InceptionV3_model():\n",
    "     #base_model = InceptionV3(weights=None, include_top=False, input_shape=(224, 224, 3))  # Changed to ResNet101\n",
    "\n",
    "#     # Add new layers on top of the model\n",
    "     #x = base_model.output  # Output of the base model\n",
    "     #x = GaussianNoise(0.1)(x)  # Add Gaussian noise with a standard deviation of 0.1\n",
    "     #x = GlobalAveragePooling2D()(x)  # Global average pooling layer\n",
    "     #x = Dense(1024, activation='relu', kernel_initializer=GlorotUniform(), kernel_regularizer=l1_l2(l1=0.02, l2=0.04))(x)  # Dense layer with ReLU activation and L2 regularization\n",
    "     #x = Dropout(0.4)(x)  # Dropout layer for regularization\n",
    "     #predictions = Dense(2, activation='softmax')(x)  # Output layer with softmax activation for multi-class classification\n",
    "\n",
    "#     # Define the model\n",
    "     #model = Model(inputs=base_model.input, outputs=predictions)  # Combined model\n",
    "\n",
    "#     # Freeze base layers\n",
    "     #for layer in base_model.layers:\n",
    "    #     layer.trainable = True  # Freeze base layers for training\n",
    "\n",
    "   #  return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "33fbb4ee",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:12:45.377894Z",
     "iopub.status.busy": "2025-02-26T17:12:45.377644Z",
     "iopub.status.idle": "2025-02-26T17:12:45.382903Z",
     "shell.execute_reply": "2025-02-26T17:12:45.382270Z"
    },
    "papermill": {
     "duration": 0.011663,
     "end_time": "2025-02-26T17:12:45.384073",
     "exception": false,
     "start_time": "2025-02-26T17:12:45.372410",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def build_Sequential_model():\n",
    "    model = keras.Sequential([\n",
    "        keras.layers.Conv2D(32,(3,3), activation='relu', input_shape = (48,48,3)),\n",
    "        keras.layers.MaxPool2D((2,2)),\n",
    "        keras.layers.Dropout(0.2),\n",
    "        \n",
    "        keras.layers.Conv2D(64,(3,3), activation='relu'),\n",
    "        keras.layers.MaxPool2D((2,2)),\n",
    "        keras.layers.Dropout(0.2),\n",
    "        \n",
    "        keras.layers.Conv2D(128,(3,3), activation='relu'),\n",
    "        keras.layers.MaxPool2D((2,2)),\n",
    "        keras.layers.Dropout(0.2),\n",
    "        \n",
    "        keras.layers.Conv2D(256,(3,3), activation='relu'),\n",
    "        keras.layers.MaxPool2D((2,2)),\n",
    "        keras.layers.Dropout(0.2),\n",
    "            \n",
    "        keras.layers.Flatten(),\n",
    "        keras.layers.Dense(128, activation='relu'),\n",
    "        keras.layers.Dense(64, activation='relu'),\n",
    "         keras.layers.Dense(32, activation='relu'),\n",
    "        keras.layers.Dense(16, activation='relu'),\n",
    "        keras.layers.Dense(2, activation='softmax')  \n",
    "    ])\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "44158bf8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:12:45.394161Z",
     "iopub.status.busy": "2025-02-26T17:12:45.393950Z",
     "iopub.status.idle": "2025-02-26T17:12:45.397145Z",
     "shell.execute_reply": "2025-02-26T17:12:45.396516Z"
    },
    "papermill": {
     "duration": 0.009517,
     "end_time": "2025-02-26T17:12:45.398379",
     "exception": false,
     "start_time": "2025-02-26T17:12:45.388862",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "lr_schedule = ReduceLROnPlateau(\n",
    "    monitor='val_loss',  # Monitor validation loss\n",
    "    factor=0.1,  # Reduce learning rate by a factor of 0.1\n",
    "    patience=3,  # Wait for 3 epochs before reducing the learning rate\n",
    "    min_lr=1e-6  # Minimum learning rate\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "0cc4ec47",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:12:45.408287Z",
     "iopub.status.busy": "2025-02-26T17:12:45.408082Z",
     "iopub.status.idle": "2025-02-26T17:12:45.411188Z",
     "shell.execute_reply": "2025-02-26T17:12:45.410564Z"
    },
    "papermill": {
     "duration": 0.009249,
     "end_time": "2025-02-26T17:12:45.412273",
     "exception": false,
     "start_time": "2025-02-26T17:12:45.403024",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Define callbacks\n",
    "early_stopping = EarlyStopping(\n",
    "    monitor='val_loss',\n",
    "    patience=5,  # Stop training if no improvement for 5 epochs\n",
    "    restore_best_weights=True\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "76900b2a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:12:45.422184Z",
     "iopub.status.busy": "2025-02-26T17:12:45.421974Z",
     "iopub.status.idle": "2025-02-26T17:12:47.832581Z",
     "shell.execute_reply": "2025-02-26T17:12:47.831723Z"
    },
    "papermill": {
     "duration": 2.416932,
     "end_time": "2025-02-26T17:12:47.833921",
     "exception": false,
     "start_time": "2025-02-26T17:12:45.416989",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.10/dist-packages/keras/src/layers/convolutional/base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
      "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1mModel: \"sequential\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓\n",
       "┃<span style=\"font-weight: bold\"> Layer (type)                         </span>┃<span style=\"font-weight: bold\"> Output Shape                </span>┃<span style=\"font-weight: bold\">         Param # </span>┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩\n",
       "│ conv2d (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">46</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">46</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)          │             <span style=\"color: #00af00; text-decoration-color: #00af00\">896</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)         │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">23</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">23</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)          │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">23</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">23</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)          │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">21</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">21</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)          │          <span style=\"color: #00af00; text-decoration-color: #00af00\">18,496</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">10</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">10</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)          │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)                  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">10</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">10</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)          │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">8</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">8</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)           │          <span style=\"color: #00af00; text-decoration-color: #00af00\">73,856</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)           │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)                  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)           │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)           │         <span style=\"color: #00af00; text-decoration-color: #00af00\">295,168</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d_3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)           │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout_3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)                  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)           │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ flatten (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Flatten</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">256</span>)                 │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                        │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)                 │          <span style=\"color: #00af00; text-decoration-color: #00af00\">32,896</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)                  │           <span style=\"color: #00af00; text-decoration-color: #00af00\">8,256</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)                  │           <span style=\"color: #00af00; text-decoration-color: #00af00\">2,080</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">16</span>)                  │             <span style=\"color: #00af00; text-decoration-color: #00af00\">528</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_4 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">2</span>)                   │              <span style=\"color: #00af00; text-decoration-color: #00af00\">34</span> │\n",
       "└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘\n",
       "</pre>\n"
      ],
      "text/plain": [
       "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓\n",
       "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                        \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape               \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m        Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩\n",
       "│ conv2d (\u001b[38;5;33mConv2D\u001b[0m)                      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m46\u001b[0m, \u001b[38;5;34m46\u001b[0m, \u001b[38;5;34m32\u001b[0m)          │             \u001b[38;5;34m896\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d (\u001b[38;5;33mMaxPooling2D\u001b[0m)         │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m23\u001b[0m, \u001b[38;5;34m23\u001b[0m, \u001b[38;5;34m32\u001b[0m)          │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout (\u001b[38;5;33mDropout\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m23\u001b[0m, \u001b[38;5;34m23\u001b[0m, \u001b[38;5;34m32\u001b[0m)          │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_1 (\u001b[38;5;33mConv2D\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m21\u001b[0m, \u001b[38;5;34m21\u001b[0m, \u001b[38;5;34m64\u001b[0m)          │          \u001b[38;5;34m18,496\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d_1 (\u001b[38;5;33mMaxPooling2D\u001b[0m)       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m10\u001b[0m, \u001b[38;5;34m10\u001b[0m, \u001b[38;5;34m64\u001b[0m)          │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout_1 (\u001b[38;5;33mDropout\u001b[0m)                  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m10\u001b[0m, \u001b[38;5;34m10\u001b[0m, \u001b[38;5;34m64\u001b[0m)          │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_2 (\u001b[38;5;33mConv2D\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m8\u001b[0m, \u001b[38;5;34m8\u001b[0m, \u001b[38;5;34m128\u001b[0m)           │          \u001b[38;5;34m73,856\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d_2 (\u001b[38;5;33mMaxPooling2D\u001b[0m)       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4\u001b[0m, \u001b[38;5;34m4\u001b[0m, \u001b[38;5;34m128\u001b[0m)           │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout_2 (\u001b[38;5;33mDropout\u001b[0m)                  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4\u001b[0m, \u001b[38;5;34m4\u001b[0m, \u001b[38;5;34m128\u001b[0m)           │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_3 (\u001b[38;5;33mConv2D\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m2\u001b[0m, \u001b[38;5;34m2\u001b[0m, \u001b[38;5;34m256\u001b[0m)           │         \u001b[38;5;34m295,168\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d_3 (\u001b[38;5;33mMaxPooling2D\u001b[0m)       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m1\u001b[0m, \u001b[38;5;34m1\u001b[0m, \u001b[38;5;34m256\u001b[0m)           │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout_3 (\u001b[38;5;33mDropout\u001b[0m)                  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m1\u001b[0m, \u001b[38;5;34m1\u001b[0m, \u001b[38;5;34m256\u001b[0m)           │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ flatten (\u001b[38;5;33mFlatten\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m256\u001b[0m)                 │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense (\u001b[38;5;33mDense\u001b[0m)                        │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m128\u001b[0m)                 │          \u001b[38;5;34m32,896\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_1 (\u001b[38;5;33mDense\u001b[0m)                      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m64\u001b[0m)                  │           \u001b[38;5;34m8,256\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_2 (\u001b[38;5;33mDense\u001b[0m)                      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m32\u001b[0m)                  │           \u001b[38;5;34m2,080\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_3 (\u001b[38;5;33mDense\u001b[0m)                      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m16\u001b[0m)                  │             \u001b[38;5;34m528\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_4 (\u001b[38;5;33mDense\u001b[0m)                      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m2\u001b[0m)                   │              \u001b[38;5;34m34\u001b[0m │\n",
       "└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">432,210</span> (1.65 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m432,210\u001b[0m (1.65 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">432,210</span> (1.65 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m432,210\u001b[0m (1.65 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import keras\n",
    "# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n",
    "# Set the learning rate\n",
    "learning_rate = 1e-4\n",
    "\n",
    "model = build_Sequential_model()\n",
    "model.summary()\n",
    "# Compile the model with the specified learning rate\n",
    "model.compile(optimizer=Adam(learning_rate=learning_rate), \n",
    "              loss='binary_crossentropy', \n",
    "              metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "cdbbec24",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:12:47.845633Z",
     "iopub.status.busy": "2025-02-26T17:12:47.845405Z",
     "iopub.status.idle": "2025-02-26T17:31:37.871726Z",
     "shell.execute_reply": "2025-02-26T17:31:37.870698Z"
    },
    "papermill": {
     "duration": 1130.033698,
     "end_time": "2025-02-26T17:31:37.873171",
     "exception": false,
     "start_time": "2025-02-26T17:12:47.839473",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.10/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:122: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.\n",
      "  self._warn_if_super_not_called()\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m499/499\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m585s\u001b[0m 1s/step - accuracy: 0.5504 - loss: 1.2335 - val_accuracy: 0.7253 - val_loss: 0.5517 - learning_rate: 1.0000e-04\n",
      "Epoch 2/50\n",
      "\u001b[1m  1/499\u001b[0m \u001b[37m━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[1m7s\u001b[0m 15ms/step - accuracy: 0.7422 - loss: 0.4964"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/lib/python3.10/contextlib.py:153: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.\n",
      "  self.gen.throw(typ, value, traceback)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m499/499\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m2s\u001b[0m 3ms/step - accuracy: 0.7422 - loss: 0.4964 - val_accuracy: 0.7119 - val_loss: 0.5588 - learning_rate: 1.0000e-04\n",
      "Epoch 3/50\n",
      "\u001b[1m499/499\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m267s\u001b[0m 529ms/step - accuracy: 0.7293 - loss: 0.5438 - val_accuracy: 0.6232 - val_loss: 0.5857 - learning_rate: 1.0000e-04\n",
      "Epoch 4/50\n",
      "\u001b[1m499/499\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 609us/step - accuracy: 0.7500 - loss: 0.5381 - val_accuracy: 0.6610 - val_loss: 0.5530 - learning_rate: 1.0000e-04\n",
      "Epoch 5/50\n",
      "\u001b[1m499/499\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m273s\u001b[0m 541ms/step - accuracy: 0.7839 - loss: 0.4710 - val_accuracy: 0.5842 - val_loss: 0.6194 - learning_rate: 1.0000e-05\n",
      "Epoch 6/50\n",
      "\u001b[1m499/499\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 644us/step - accuracy: 0.7109 - loss: 0.5363 - val_accuracy: 0.6695 - val_loss: 0.5723 - learning_rate: 1.0000e-05\n"
     ]
    }
   ],
   "source": [
    "# Train the model using the generators\n",
    "hist = model.fit(\n",
    "    train_generator,  # Use the train_generator created with flow_from_dataframe\n",
    "    steps_per_epoch=len(train_data_df) // batch_size,  # Number of batches per epoch\n",
    "    epochs=epochs,\n",
    "    validation_data=val_generator,  # Use the val_generator created with flow_from_dataframe\n",
    "    validation_steps=len(val_data_df) // batch_size,  # Number of validation batches\n",
    "    callbacks=[early_stopping, lr_schedule],  # Callbacks for early stopping and learning rate scheduling\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "18cc341e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:31:38.021269Z",
     "iopub.status.busy": "2025-02-26T17:31:38.020992Z",
     "iopub.status.idle": "2025-02-26T17:31:38.078617Z",
     "shell.execute_reply": "2025-02-26T17:31:38.077723Z"
    },
    "papermill": {
     "duration": 0.131851,
     "end_time": "2025-02-26T17:31:38.080019",
     "exception": false,
     "start_time": "2025-02-26T17:31:37.948168",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "model.save('model_Sequential_run1.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "39ef7a48",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:31:38.227078Z",
     "iopub.status.busy": "2025-02-26T17:31:38.226795Z",
     "iopub.status.idle": "2025-02-26T17:35:01.379457Z",
     "shell.execute_reply": "2025-02-26T17:35:01.378771Z"
    },
    "papermill": {
     "duration": 203.227541,
     "end_time": "2025-02-26T17:35:01.380903",
     "exception": false,
     "start_time": "2025-02-26T17:31:38.153362",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m44/44\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m191s\u001b[0m 4s/step\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "# Make predictions using the test generator\n",
    "predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)\n",
    "\n",
    "# Convert predictions to class labels (assuming 2 classes)\n",
    "predicted_labels = np.argmax(predictions, axis=1)  # Get the index of the highest probability\n",
    "\n",
    "# Add predictions to the test DataFrame\n",
    "test_df['label'] = predicted_labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "079e1703",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:35:01.539653Z",
     "iopub.status.busy": "2025-02-26T17:35:01.539387Z",
     "iopub.status.idle": "2025-02-26T17:35:01.542339Z",
     "shell.execute_reply": "2025-02-26T17:35:01.541659Z"
    },
    "papermill": {
     "duration": 0.083377,
     "end_time": "2025-02-26T17:35:01.543668",
     "exception": false,
     "start_time": "2025-02-26T17:35:01.460291",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# # Make predictions using the test generator\n",
    "# predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)\n",
    "\n",
    "# # Convert predictions to binary labels\n",
    "# predicted_labels = [1 if pred > 0.5 else 0 for pred in predictions]\n",
    "\n",
    "# # Add predictions to the test DataFrame\n",
    "# test_df['label'] = predicted_labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "ce96c5cf",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:35:01.694856Z",
     "iopub.status.busy": "2025-02-26T17:35:01.694603Z",
     "iopub.status.idle": "2025-02-26T17:35:01.699132Z",
     "shell.execute_reply": "2025-02-26T17:35:01.698355Z"
    },
    "papermill": {
     "duration": 0.081157,
     "end_time": "2025-02-26T17:35:01.700327",
     "exception": false,
     "start_time": "2025-02-26T17:35:01.619170",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Create submission DataFrame\n",
    "submission_df = pd.DataFrame({\n",
    "    'id': test_df['id'],\n",
    "    'label': test_df['label'].astype(int)  # Convert predictions to integers\n",
    "})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "5b04674c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-02-26T17:35:01.850808Z",
     "iopub.status.busy": "2025-02-26T17:35:01.850599Z",
     "iopub.status.idle": "2025-02-26T17:35:01.866681Z",
     "shell.execute_reply": "2025-02-26T17:35:01.866012Z"
    },
    "papermill": {
     "duration": 0.092456,
     "end_time": "2025-02-26T17:35:01.867882",
     "exception": false,
     "start_time": "2025-02-26T17:35:01.775426",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Submission file 'submission.csv' created successfully.\n"
     ]
    }
   ],
   "source": [
    "# # Save the submission file, remove rows with NaN labels\n",
    "# submission_df = submission_df.dropna(subset=['label'])\n",
    "submission_df.to_csv('submission_sh_Sequential.csv', index=False)\n",
    "print(\"Submission file 'submission.csv' created successfully.\")"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "nvidiaTeslaT4",
   "dataSources": [
    {
     "datasetId": 6412205,
     "sourceId": 10550636,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 30918,
   "isGpuEnabled": true,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 1456.100386,
   "end_time": "2025-02-26T17:35:03.766417",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2025-02-26T17:10:47.666031",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
