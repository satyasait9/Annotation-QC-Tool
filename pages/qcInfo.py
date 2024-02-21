
import pages.utilities as util
ss=util.st.session_state
################################
#######   PAGE CONFIG ##########
################################

util.st.cache_data.clear()

util.st.set_page_config(page_title="QC Information",
                   initial_sidebar_state="collapsed",
                   page_icon="🧊")
#SideBar Display To Switch Pages
util.show_pages_from_config()
util.hide_pages(["Admin Dashboard","QC Information", "QC Tool"])

def startQC():
    ss.startQC = not ss.startQC

#########################################
#######SESSION STATE VARIABLES###########
#########################################

# Function to switch between True/False when button is pressed
if "startQC" not in ss:
    ss.startQC = False

# Global variables
if "dataset_date" not in ss:
    ss.dataset_date = None

if "label_folder_path" not in ss:
    ss.label_folder_path = None

if "image_folder_path" not in ss:
    ss.image_folder_path = None

if "image_list" not in ss:
    ss.image_list = None

if "qc_image_list" not in ss:
    ss.qc_image_list = None

if "split_id" not in ss:
    ss.split_id = None

if "length_of_qc_list" not in ss:
    ss.length_of_qc_list = None

if "qc_per" not in ss:
    ss.qc_per = None

if "stat_image_list" not in ss:
    ss.stat_image_list = None

# Dataframe 
if "df" not in ss:
    ss.df = None

if "img_quality_list" not in ss:
    ss.img_quality_list = None

if "remarks_list" not in ss:
    ss.remarks_list = None

if "tp_lst" not in ss:
    ss.tp_lst = None

if "fp_list" not in ss:
    ss.fp_list = None

if "tn_list" not in ss:
    ss.tn_list = None
if "fn_list" not in ss:
    ss.fn_list = None
if "no_of_ann" not in ss:
    ss.no_of_ann = None
if "csv_data" not in ss:
    ss.csv_data=None
if 'dataset_type' not in ss:
    ss.dataset_type = None
if "image_dataset" not in ss:
    ss.image_dataset=None
if "counter" in ss:
    ss.counter=0
if "label_color_dict" not in ss:
    ss.label_color_dict = None

################################
#######      UI      ###########
################################

# Title
col1, col2 = util.st.columns([1,20])
with col1:
    util.st.image('assets/april_logo.png', width=60)
with col2:
    util.st.markdown("# <center>Annotation QC Tool!</center> ", unsafe_allow_html=True)
util.st.divider()

if ss.valid_user:
    util.st.cache_data.clear()

    # Date
    todays_date = util.datetime.datetime.now()
    jan_1 = util.datetime.date(todays_date.year, 1, 1)
    ss.dataset_date = todays_date
    
    # Dataset name input
    user_id = util.get_person_id(ss.username)
    project_exists = util.check_project_existance(user_id)
    # util.st.write(project_exists)
    if project_exists:
        datasetName = util.get_dataset_values(user_id)
        data= util.st.selectbox("Dataset", options=datasetName, index=None)
        if "dataset_name" not in ss:
            ss.dataset_name = None
        if data != None:
            ss.dataset_name = data
            #Getting Respective Id's
            project_id=util.get_dataset_id(ss.dataset_name)
            split_id=util.generate_split_id(project_id, user_id)

            ss.split_id = split_id

            # Image and Label folder paths selection
            valid_image_path, valid_label_path = False, False
            image_folder_path, label_folder_path = util.get_paths(ss.dataset_name)

            ss.image_folder_path = image_folder_path

            # label_folder_path = get_label_folder(ss.dataset_name)
            ss.label_folder_path = label_folder_path

            util.st.button(label="Start QC", use_container_width=True, on_click=startQC)

            # Assign Dataframe
            ss.df = util.get_df(split_id)
            # Assign values to session state variables
            ss.qc_image_list = ss.df['Image'].values
            ss.length_of_qc_list = len(ss.qc_image_list)
            ss.image_list = ss.df['Image'].values
            ss.stat_image_list = [f for f in util.os.listdir(ss.image_folder_path) if f.endswith(('.png', '.jpg'))]
            ss.qc_per = util.percentage_of_qc_images(ss.stat_image_list, ss.qc_image_list)
            
            ss.img_quality_list = ss.df['Image Quality'].values.tolist()
            ss.remarks_list = ss.df['Remarks'].values.tolist()
            ss.tp_list = ss.df['True_Positive'].values.tolist()
            ss.fp_list = ss.df['False_Positive'].values.tolist()
            ss.tn_list = ss.df['True_Negative'].values.tolist()
            ss.fn_list = ss.df['False_Negative'].values.tolist()
            ss.no_of_ann=ss.df['No_of_Objects_Annotated'].values.tolist

            # Once all of this is done then switch page
            if ss.startQC and util.st.button:
                util.st.switch_page("pages/qcMain.py")
    else:
        util.st.info("Project not assigned!!")
                    
