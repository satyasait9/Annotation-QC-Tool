# Libraries
import os
import io
import cv2
import random
import datetime
import psycopg2
import subprocess
import numpy as np
import pandas as pd
from io import BytesIO
import streamlit as st
from pathlib import Path
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import xml.etree.ElementTree as ET
from multiprocessing.pool import ThreadPool
from st_pages import show_pages_from_config, hide_pages

ss=st.session_state
# Database configuration
database_name = 'db'
user = 'postgres'
password = '12345678'
host = 'localhost'
port = '5432'

# Database connection
connection_params = {
        "database": f'{database_name}',
        "user": f'{user}',
        "password": f'{password}',
        "host": f'{host}',
        "port": f'{port}'
    }

def clear():
    ss.clear()
#####data type of dataset######
def fetch_dataset_type(dataset):
    with psycopg2.connect(**connection_params) as conn,conn.cursor() as cur:
        query = f"select dataset_type from project where dataset='{dataset}'"
        cur.execute(query)
        datatype = cur.fetchone()[0]
    return datatype


##### Login Check  #####
def read_user_credentials(user, password):
    with psycopg2.connect(**connection_params) as conn,conn.cursor() as cur:
        query = f"SELECT EXISTS(SELECT 1 FROM user_table WHERE person='{user}')"
        cur.execute(query)
        exist = cur.fetchone()[0]
        if exist:
            query = f"SELECT 1 FROM user_table WHERE person='{user}' AND password='{password}'"
            cur.execute(query)
            ans = cur.fetchone()
            if ans:
                return True
        else:
            return False

#### save the DF  ####
def save_user_progress(split_id, csv_data):
    with psycopg2.connect(**connection_params) as conn,conn.cursor() as cur:
        # Assuming "work" is the table name and "files" is the column name
        # Use psycopg2.Binary to convert CSV data to binary format
        query = f"UPDATE work SET files = {psycopg2.Binary(csv_data)} WHERE split_id = '{split_id}'"
        # Execute the query with parameters
        cur.execute(query)
        # Commit the changes to the database
        conn.commit()

#### Get the individual DataFrame by user_id and project_id ####
def get_user_dataframe(pid,uid):
    query = f"SELECT files FROM work WHERE project_id={pid} AND user_id={uid}"
    with psycopg2.connect(**connection_params) as conn,conn.cursor() as cur:
        cur.execute(query)
        a = cur.fetchone()[0]
    return pd.read_csv(io.BytesIO(a))

def df_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

#Function for project_id from project table by project name
def get_dataset_id(dataset):
    query = f"SELECT project_id FROM project WHERE dataset = '{dataset}'"
    with psycopg2.connect(**connection_params) as conn,conn.cursor() as cur:
        cur.execute(query)
        result = cur.fetchone()
    return result[0] if result else None

#Function for user_id from user_table by person name
def get_person_id(qc):
    query = f"SELECT user_id FROM user_table WHERE person = '{qc}'"
    with psycopg2.connect(**connection_params) as conn,conn.cursor() as cur:
        cur.execute(query, (qc,))
        result = cur.fetchone()
    return result[0] if result else None

#Function to get all names of persons in uer_table as list 
def get_QC_users():
    query = f"SELECT person FROM user_table"
    with psycopg2.connect(**connection_params) as conn ,conn.cursor() as cur:
        cur.execute(query)
        column_values = [value[0] for value in cur.fetchall()]
    return column_values

#Function to get all project names in project table as list 
def get_all_dataset():
    query = f"SELECT dataset FROM project"
    with psycopg2.connect(**connection_params) as conn,conn.cursor() as cur:
        cur.execute(query)
        column_values = [value[0] for value in cur.fetchall()]
    return column_values

#Function to check login by credentials
def check_login(user, password):
    if user and password:
        #First Checks wherther the user exists
        query = f"SELECT EXISTS(SELECT 1 FROM user_table WHERE person='{user}')"
        with psycopg2.connect(**connection_params) as conn,conn.cursor() as cur:
            cur.execute(query, (user,))
            exist = cur.fetchone()[0]
        if exist:
            st.error("User already exists", icon='❌')
        else:
            #if user exists checks the password to corresponding user
            query = f"INSERT INTO user_table(person, password) VALUES ('{user}', '{password}')"
            with psycopg2.connect(**connection_params) as conn,conn.cursor() as cur:
                cur.execute(query)
                conn.commit()
                st.write("User created successfully", icon="✅")
    else:
        st.write("Fill Details")
#Function to refresh file data in project table
def refresh_Dataset():
    with psycopg2.connect(**connection_params) as conn,conn.cursor() as cur:
        #collects all project_id from work table
        query = "SELECT DISTINCT project_id FROM work"
        cur.execute(query)
        project_ids = [row[0] for row in cur.fetchall()]
        for project_id in project_ids:
            #check whether the project_id exist in project table
            query_exists = f"SELECT EXISTS(SELECT 1 FROM work WHERE project_id={project_id})"
            cur.execute(query_exists)
            exists = cur.fetchone()[0]
            if exists:
                #takes the individual file data from every user who has been assinged this project_id
                query_select = f"SELECT files FROM work WHERE project_id = {project_id}"
                cur.execute(query_select)
                all_csv = cur.fetchall()
                #takes all as csv and conacates by pandas 
                final_df = pd.concat([pd.read_csv(io.BytesIO(binary_data[0])) for binary_data in all_csv], axis=0, ignore_index=True, sort=False)
                #dataframe to csv
                csv_bytes = final_df.to_csv(index=False).encode()
                #csv to binary and stores it in file_fdata of project table
                query_update = f"UPDATE project SET file_data = {psycopg2.Binary(csv_bytes)} WHERE project_id = {project_id}"
                cur.execute(query_update)
                conn.commit()

#function to check whether dataset exist or not by project name
def dataset_exists(dataset):
    with psycopg2.connect(**connection_params) as conn,conn.cursor() as cur:
        query = f"SELECT 1 FROM project WHERE dataset = '{dataset}'"
        cur.execute(query)
        exist = cur.fetchone()
    return exist

#Function to display the whole project file as Dataframe
def display_dataset(dataset):
    with psycopg2.connect(**connection_params) as conn,conn.cursor() as cur:
        query = f"SELECT file_data FROM project WHERE dataset='{dataset}'"
        cur.execute(query)
        binary_file = cur.fetchone()
    if binary_file:
        df = pd.read_csv(BytesIO(binary_file[0]))
        return df
    else:
        return None
    
#Function to get all the users who have assigned the the particular project_id as list
def user_names(dataset):
    with psycopg2.connect(**connection_params) as conn,conn.cursor() as cur:
        project_id = get_dataset_id(dataset)
        query = f"SELECT user_id FROM work WHERE project_id='{project_id}'"
        cur.execute(query)
        user_ids = [row[0] for row in cur.fetchall()]
        names = []
        for user_id in user_ids:
            query_user = f"SELECT person FROM user_table WHERE user_id='{int(user_id)}'"
            cur.execute(query_user)
            name = cur.fetchone()[0]
            names.append(name)
    return names
#Function to Create a dataset by taking image and label folder paths and type of dataset(like dota or pascal_voc)
def create_dataset(dataset, image_folder, label_folder, qc_per, dataset_type):
    if dataset_exists(dataset):
        st.warning(f"The dataset '{dataset}' already exists.")
    else:
        qc_per = int(qc_per)
        if 0 < qc_per <= 100:
            image_list = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
            num_samples = int(len(image_list) * qc_per / 100)
            qc_list = sorted(random.sample(image_list, num_samples))

            df = pd.DataFrame({
                "Image": qc_list,
                "Image Quality": [None] * len(qc_list),
                "Remarks": [None] * len(qc_list),
                "No_of_Objects_Annotated": [single_image_ann_info(image, label_folder, dataset_type) for image in qc_list],
                "True_Positive": [0] * len(qc_list),
                "False_Positive": [0] * len(qc_list),
                "True_Negative": [0] * len(qc_list),
                "False_Negative":[0] * len(qc_list)
            })

            CSV = df.to_csv(index=False).encode("utf-8")

            # Perform dataset creation
            insert_query = f"INSERT INTO project (dataset, image_folder, label_folder, file_data, list, dataset_type) VALUES ('{dataset}', '{image_folder}', '{label_folder}', {psycopg2.Binary(CSV)}, {len(qc_list)}, '{dataset_type}')"
            
            with psycopg2.connect(**connection_params) as conn,conn.cursor() as cur:
                cur.execute(insert_query)
                conn.commit()
            st.success(f"The dataset '{dataset}' was created successfully.")
        else:
            st.write("Enter a valid percentage")
#To Split the dataset into number of users selected
def assign_dataset(data, qcusers):
    project_id = get_dataset_id(data)
    with psycopg2.connect(**connection_params) as conn,conn.cursor() as cur:
        query_exists = f"SELECT EXISTS(SELECT 1 FROM work WHERE project_id='{project_id}')"
        cur.execute(query_exists)
        exists = cur.fetchone()[0]
        if exists:
          st.write("Dataset already splitted")
        else:
            query_select = "SELECT file_data FROM project WHERE dataset = %s"
            cur.execute(query_select, (data,))
            binary_file = cur.fetchall()[0][0]
            df = pd.read_csv(BytesIO(binary_file))
            a = len(df["Image"])
            if a / len(qcusers) > 0:
                parts = np.array_split(df, len(qcusers))
                name_list = []
                for i, part in enumerate(parts, 0):
                    user_id = get_person_id(qcusers[i])
                    name_list.append(qcusers[i])
                    split_id = datetime.datetime.today().date().strftime('%d%m%Y') + data + qcusers[i]
                    CSV = part.to_csv(index=False).encode("utf-8")
                    query_insert = f"INSERT INTO work(project_id, user_id, split_id, files) VALUES ({project_id}, {user_id}, '{split_id}', {psycopg2.Binary(CSV)})"
                    cur.execute(query_insert)
                    conn.commit()
                st.success("Dataset successfully assigned to users.")
            else:
                st.write("Decrease users")



# Database connection function
def connect_db():
    return psycopg2.connect(database=database_name, user=user, password=password, host=host, port=port)

# Function to execute a query and fetch one value
def execute_fetch_one(query, params=None):
    with connect_db() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchone()[0] if cur.rowcount else None


def get_dataset_values(user_id):
    query = f"SELECT project_id FROM work WHERE user_id='{user_id}'"
    project_ids = execute_fetch_all(query)
    return [execute_fetch_one("SELECT dataset FROM project WHERE project_id=%s", (pid,)) for pid in project_ids]


# Function to generate split id
def generate_split_id(project_id, user):
    query = f"SELECT split_id FROM work WHERE project_id='{project_id}' AND user_id='{user}'"
    return execute_fetch_one(query)

# Function to get dataset id
def get_dataset_id(dataset):
    query = f"SELECT project_id FROM project WHERE dataset = '{dataset}'"
    return execute_fetch_one(query)

# Function to get person id
def get_person_id(qc):
    query = f"SELECT user_id FROM user_table WHERE person = '{qc}'"
    return execute_fetch_one(query)

# Function to get DataFrame
def get_df(split):
    query = f"SELECT files FROM work WHERE split_id='{split}'"
    with connect_db() as conn, conn.cursor() as cur:
        cur.execute(query)
        binary_file = cur.fetchall()[0][0]
    return pd.read_csv(BytesIO(binary_file))

# Function to execute a query and fetch all values
def execute_fetch_all(query, params=None):
    with connect_db() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()
#Funtion to get image and label folder paths by dataset name  
def get_paths(dataset):
    query = f"SELECT (image_folder, label_folder) FROM project WHERE dataset='{dataset}'"
    with connect_db() as conn, conn.cursor() as cur:
        cur.execute(query)
        split_paths = str(cur.fetchall()[0][0]).split(',')
        image_path, label_path = split_paths[0][1:], split_paths[-1][:-1]
        image_path=image_path.replace('"','')
        label_path=label_path.replace('"','')
        return image_path, label_path
#Function to check whether a project has benn assigned to user or not  
def check_project_existance(user_id):
    query = f"SELECT exists(SELECT 1 FROM work WHERE user_id = '{user_id}')"
    with connect_db() as conn, conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchone()[0]


################################
#######   FUNCTIONS  ###########
################################

#Function for qc percwntage of images to be taken
def percentage_of_qc_images(image_list, qc_image_list):
    common_images = set(image_list) & set(qc_image_list)
    if not common_images:
        return 0.0  # No common images
    percentage = (len(common_images) / len(image_list)) * 100
    return percentage




################################
#######   FUNCTIONS  ###########
################################

# Function to extract image size
def load_image(image_path):
    return cv2.imread(image_path).shape[:2]

#Funtion to get image shape
def image_size(image_folder_path, image_list):
    img_shape = set()
    pool = ThreadPool()
    image_paths = [os.path.join(image_folder_path, img) for img in image_list]
    results = pool.map(load_image, image_paths)
    pool.close()
    pool.join()
    img_shape.update(results)
    return img_shape

# Function to extract annotation information
def label_info(image_list, label_folder_path, dataset_type):
    '''
    This function takes a list of image filenames as input and returns information 
    about the labels associated with those images.

    Parameters:
    - image_list (list): A list of image filenames.

    Returns:
    - classes (set): A set of unique class names present in the labels.
    - total_annotations (int): The total number of annotations across all labels.
    - class_counts (dict): A dictionary containing the count of each class in the labels.
    '''
    
    label_list = sorted([f[:f.rfind('.')] for f in image_list])

    total_annotations = 0
    class_counts = {}
    classes = set()

    for labels in label_list:
        if dataset_type == "DOTA":
            filepath = os.path.join(label_folder_path, labels + '.txt')
            if os.path.exists(filepath):
                with open(filepath, 'r') as file:
                    annotations = file.readlines()
                    for annotation in annotations:
                        annotation = annotation.strip().split()
                        if len(annotation) < 9:
                            continue  # Skip if the annotation does not have enough values

                        class_name = annotation[8]
                        total_annotations += 1

                        # Update class count in the dictionary
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                classes.update(class_counts.keys())
            else:
                continue

        elif dataset_type == "PASCAL_VOC":
            filepath = os.path.join(label_folder_path, labels + '.xml')
            if os.path.exists(filepath):
                try:
                    # Parse the XML label file
                    tree = ET.parse(filepath)
                    root = tree.getroot()
                    
                    # Check if the XML file is empty
                    if len(root) == 0:
                        continue

                    else:
                        for obj in root.findall('object'):
                            class_name = obj.find('name').text
                            total_annotations += 1

                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        classes.update(class_counts.keys())
                except ET.ParseError:
                    continue

    return list(classes), int(total_annotations), class_counts

# Function to extract the no. of annotations in single image

def single_image_ann_info(image_path, label_folder_path, dataset_type):
    '''
    This function calculates the number of annotations present in a given image.

    Parameters:
    - image_path (str): The path of the image file.

    Returns:
    - no_of_annotations (int): The number of annotations in the image.
    '''

    no_of_annotations = 0

    if dataset_type == "DOTA":
        filename = image_path[:image_path.rfind('.')] + '.txt'
        filepath = os.path.join(label_folder_path, filename) 
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                annotations = f.readlines()
                if len(annotations) < 1:
                    no_of_annotations = 0
                for annotation in annotations:
                    if len(annotation.strip()) > 0:  # Check for non-empty lines
                        no_of_annotations += 1
        else: 
            no_of_annotations = 0
    
    if dataset_type == "PASCAL_VOC":
        filename = image_path[:image_path.rfind('.')] + '.xml'
        filepath = os.path.join(label_folder_path, filename)
        if os.path.exists(filepath):
            try:
                # Parse the XML label file
                tree = ET.parse(filepath)
                root = tree.getroot()
                
                # Check if the XML file is empty
                if len(root) == 0:
                    no_of_annotations = 0
                else:
                    # Iterate over each object in the annotation
                    no_of_annotations = len(root.findall('object'))

            except ET.ParseError:
                no_of_annotations = 0

    return no_of_annotations

#Function for Vector_TransForm in xml bound and rbound
def to_vector_transforms(cosVal, sinVal, corners, cx, cy):
    newCorners = []
    for x,y in corners:
        x0 = cx-((x-cx)*cosVal + (y-cy)*sinVal)
        y0 = cy-(x-cx)*sinVal + (y-cy)*cosVal
        newCorners.append((x0,y0))
    return newCorners

#Function to Find Corners in xml bound and rbound
def find_corners(cx, cy, w, h, angle):
  """
  Finds the four corners of a box given the x_centre, y_centre, width, height, and angle.

  Args:
    x_centre: The x-coordinate of the centre of the box.
    y_centre: The y-coordinate of the centre of the box.
    width: The width of the box.
    height: The height of the box.
    angle: The angle of the box in degrees.

  Returns:
    A list of four points representing the corners of the box.
  """

  # Convert the angle to radians.
#   angle = angle * math.pi / 180
  sin_angle=np.sin(angle)
  cos_angle=np.cos(angle)
  # Calculate the x- and y-coordinates of the four corners.
  corners = [(int(cx-w/2),int(cy-h/2)),(int(cx+w/2),int(cy-h/2)),(int(cx+w/2),int(cy+h/2)),(int(cx-w/2),int(cy+h/2))]             

  # Rotate the corners by the specified angle.
  box = to_vector_transforms(cos_angle,sin_angle,corners,cx,cy)
  # Return the corners as a list of points.
  return box

# Function to generate colors for each class
def generate_color(label_list: list):
    # Generate random colors for each class
    label_color_dict: dict = {}
    
    for label in label_list:
        # Generate random RGB values
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        label_color_dict[label] = color

    return label_color_dict


#Function TO Annotate Image By txt or xml 
def annotate_image(image_path: str, label_folder_path: str, dataset_type: str,label_color_dict: dict, view_labels: bool = True):
    '''
    This function takes an image path and a label folder path as input and annotates the 
    image with bounding boxes and labels based on the annotations present in the label file.

    Parameters:
    - image_path (str): The path of the image file.
    - label_folder_path (str): The path of the folder containing the label files.
    - view_labels (bool, optional): Whether to display the labels on the annotated image. Defaults to True.

    Returns:
    - annotated_image (numpy.ndarray): The annotated image with bounding boxes and labels.
    '''
    # Read the image file
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid or non-existent image file")

    # Convert image to RGB if it has 4 channels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[2] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract the filename from the image path
    filename = Path(image_path).name
    is_negative: bool = False
    
    if dataset_type == "DOTA":
        # Construct the label file path
        label_path = Path(label_folder_path) / (filename[:filename.rfind('.')] + '.txt')

        if os.path.exists(label_path):
            is_negative = False
            
            # Open the label file in read mode
            with open(label_path, 'r') as file:
                for annotation in file:
                    annotation = annotation.strip().split()

                    # Skip if the annotation does not have enough values
                    if len(annotation) < 1:
                        is_negative = True
                        continue

                    # Extract the label
                    label = annotation[8]

                    try:
                        # Convert the coordinates to integers after rounding the floating-point values
                        coords = np.array([[int(round(float(x))), int(round(float(y)))] for x, y in zip(annotation[:8:2], annotation[1:8:2])])
                    except ValueError:
                        continue  # Skip if the conversion to float fails

                    # Draw a polygon on the image
                    cv2.polylines(image, [coords], isClosed=True, color=label_color_dict[label])

                    # Display the label on the image if view_labels is True
                    if view_labels:
                        x1, y1 = coords[0]
                        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color_dict[label], 2)

        else:
            is_negative = True

    if dataset_type == 'PASCAL_VOC':
        # Construct the XML label file path
        label_path = Path(label_folder_path) / (filename[:filename.rfind('.')] + '.xml')

        if os.path.exists(label_path):
            try:
                # Parse the XML label file
                tree = ET.parse(label_path)
                root = tree.getroot()
                
                # Check if the XML file is empty
                if len(root) == 0:
                    is_negative = True
                # else:
                #     continue
                    # is_negative = False

                # Iterate over each object in the annotation
                for obj in root.findall('object'):
                    # Extract label
                    label = obj.find('name').text
                    st.write(label)

                    if obj.find('bndbox') is not None:

                        # Extract bounding box coordinates
                        xmin = int(obj.find('bndbox').find('xmin').text)
                        ymin = int(obj.find('bndbox').find('ymin').text)
                        xmax = int(obj.find('bndbox').find('xmax').text)
                        ymax = int(obj.find('bndbox').find('ymax').text)

                        # Convert coordinates to numpy array
                        coords = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])

                        # Draw bounding box on the image
                        cv2.polylines(image, [coords], isClosed=True, color=label_color_dict[label])

                        # Display label on the image if view_labels is True
                        if view_labels:
                            cv2.putText(image, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color_dict[label], 2)
                    

                    elif obj.find('robndbox') is not None:
                        bbox = obj.find('robndbox')
                        #print("format for bbox is : ",bbox)
                        cx = float(bbox.find('cx').text)
                        cy = float(bbox.find('cy').text)
                        w = float(bbox.find('w').text)
                        h = float(bbox.find('h').text)
                        angle = float(bbox.find('angle').text)
                        
                        # Convert the angle to radians.
                        yolo_obb_coords = find_corners(cx,cy,w,h,angle)
                        yolo_obb_coords = [(int(x), int(y)) for x, y in yolo_obb_coords] 

                        # Draw a polygon on the image
                        cv2.polylines(image, np.int32([yolo_obb_coords]), isClosed=True, color=label_color_dict[label])
                    
                        # Display the label on the image if view_labels is True
                        if view_labels:
                            x1, y1 = yolo_obb_coords[0]
                            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color_dict[label], 2)
                    
                    else:
                        is_negative = True

            except ET.ParseError:
                is_negative = True
            
        else:
            is_negative = True

    return image, is_negative


# Function to calculate accuracy
def calculate_accuracy(total_number_of_objects_annotated, incorrect_annotation):
    correct_annotations = total_number_of_objects_annotated - incorrect_annotation
    if total_number_of_objects_annotated == 0:
        return 0
    annotation_accuracy = (correct_annotations / total_number_of_objects_annotated) * 100
    return annotation_accuracy

# Function to close the server
def stop_streamlit_server():
    '''
    This function stops the Streamlit server by killing the process associated with it.
    '''

    process = subprocess.Popen(["pkill", "-f", "streamlit"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    if process.returncode == 0:
        st.success("Streamlit app stopped successfully.")
    else:
        st.error(f"Failed to stop Streamlit app. Error: {err.decode()}")
#Function to Extract classes and annotation count by image_list and folder paths
def calculate_dataset_stats(image_list, label_folder_path, dataset_name):
    classes, total_annotations, class_counts = label_info(image_list, label_folder_path, fetch_dataset_type(dataset_name))
    image_shapes = image_size(ss.image_folder_path, image_list)
    return classes, total_annotations, class_counts, image_shapes
#Function to create a BarChart for stats
def create_bar_chart(classes, class_counts, title, label_color_dict):
    filtered_label_color_dict = {key: label_color_dict[key] for key in classes if key in label_color_dict.keys()}
    colors = [f'rgb{filtered_label_color_dict[c]}' for c in filtered_label_color_dict.keys()]
    fig = px.bar(x=list(class_counts.keys()), y=list(class_counts.values()), text_auto=True,
                 labels={'x': 'Class', 'y': 'Annotation Count'}, title=title, color=filtered_label_color_dict.keys(), 
                 color_discrete_sequence=colors)
    return fig

# Function to track the QC progress of the user
def trackQCprogress(ss):
    l=[]
    totalCompleted=0
    classification_accuracy=0
    f1_score=0
    recall=0
    precision=0
    for i in range(len(ss.qc_image_list)):
        if not(pd.isnull(ss['df']['Image Quality'][i])) and not(pd.isnull(ss['df']['Remarks'][i])):
            totalCompleted+=1
            if (ss['df']['True_Positive'][i]+ss['df']['True_Negative'][i]+ss['df']['False_Positive'][i]+ss['df']['False_Negative'][i])!=0:
                classification_accuracy+=(ss['df']['True_Positive'][i]+ss['df']['True_Negative'][i])/(ss['df']['True_Positive'][i]+ss['df']['True_Negative'][i]+ss['df']['False_Positive'][i]+ss['df']['False_Negative'][i])
    l.append(totalCompleted)
    l.append(classification_accuracy)
    return l

@st.cache_data(show_spinner="Fetching stats from data...")
#Direct Function to Calculate all stats and Place it in SideBar
def fetch_dataset_stats():
    # Calculate stats for main and QC datasets
    main_classes, main_total_annotations, main_class_counts, main_image_shapes = calculate_dataset_stats(ss.stat_image_list, ss.label_folder_path, ss.dataset_name)
    qc_classes, qc_total_annotations, qc_class_counts, qc_image_shapes = calculate_dataset_stats(ss.qc_image_list, ss.label_folder_path, ss.dataset_name)

    # Config for plotly charts
    config = {'displayModeBar': False}

    # Update sidebar with dataset statistics
    st.sidebar.markdown("# <u> Main Dataset </u>", unsafe_allow_html=True)
    st.sidebar.write("## Dataset Image Shape(s):", main_image_shapes)
    st.sidebar.write("## Images & Labels: ", len(ss.stat_image_list))
    st.sidebar.write("## Total Annotations: ", main_total_annotations)
    st.sidebar.write("## Classes: ")
    for i in range(len(main_classes)):
        st.sidebar.write(main_classes[i].upper())
    st.sidebar.plotly_chart(create_bar_chart(sorted(main_classes), main_class_counts, 'Annotation Count for Each Class', label_color_dict=ss.label_color_dict), use_container_width=True, config=config)
    st.sidebar.divider()

    st.sidebar.markdown(f"# <u> {int(ss.qc_per)}% of Dataset: </u>", unsafe_allow_html=True)
    st.sidebar.write("## QC Dataset Image Shape(s):", qc_image_shapes)
    st.sidebar.write("## QC Images & Labels: ", len(ss.qc_image_list))
    st.sidebar.write("## Total Annotations: ", qc_total_annotations)
    st.sidebar.write("## Classes: ")
    for i in range(len(qc_classes)):
        st.sidebar.write(qc_classes[i].upper())
    # st.sidebar.plotly_chart(create_bar_chart(qc_classes, qc_class_counts, 'QC Annotation Count for Each Class', label_color_dict=ss.label_color_dict), use_container_width=True, config=config)
    st.sidebar.divider()

    # Pie-chart for dataset distribution
    non_qc_images = len(ss.stat_image_list) - ss.length_of_qc_list
    fig = px.pie(names=['QC Images', 'Non-QC Images'], values=[ss.length_of_qc_list, non_qc_images])
    st.sidebar.plotly_chart(fig, use_container_width=True)

#To Clear Cache
def clear_cache():
    st.cache_data.clear()

