
import pages.utilities as util
util.clear_cache()
ss=util.st.session_state
if "tp" not in ss:
    ss.tp = None

if "fp" not in ss:
    ss.fp = None

if "tn" not in ss:
    ss.tn = None
if "fn" not in ss:
    ss.fn = None
if "no_of_ann" not in ss:
    ss.no_of_ann = None
if "label_color_dict" not in ss:
    ss.label_color_dict=None

### For Changing Counter Value By Manual Typing Number ###
def update_index():
    if ss.admin_image_index != ss.admin_counter:
        ss.admin_counter = ss.admin_image_index
        
    elif ss.admin_image_index > len(ss.qc_image_list) - 1:
        ss.admin_image_index = len(ss.qc_image_list) - 1

    elif ss.admin_image_index < 0:
        ss.admin_image_index = 0

### Go Next By Clicking + ###
def onNext():
    if ss.admin_counter <= len(ss.qc_image_list) - 1:
        ss.admin_counter += 1 


### Go Back By Clicking - ###
def onPrev():
    if ss.admin_counter > 0:
        ss.admin_counter -= 1
### For Zooming Option In Image ###
def toggle_view_image():
    ss.view_image = not ss.view_image
### For ComeBack To Normal View ###
def enable_view():
    ss.view_image_status = not ss.view_image_status
    if 'admin_counter' in ss:
        ss.admin_counter = 0


# UI
util.st.set_page_config(page_title="Admin Dashboard",
                   page_icon="assets/april_logo.png",
                   initial_sidebar_state="collapsed")
util.hide_pages(["QC Information", "QC Tool","Admin Dashboard"])

# Title
col1, col2 = util.st.columns([1,20])
with col1:
    util.st.image('assets/april_logo.png', width=60)
with col2:
    util.st.markdown("# <center>Annotation QC Tool!</center> ", unsafe_allow_html=True)
util.st.divider()

### Session States ###
if "view_image_status" not in ss:
    ss.view_image_status = None

if "dataset" not in ss:
    ss.dataset = None

if "qc_per" not in ss:
    ss.qc_per = None

if "view_image" not in ss:
    ss.view_image = False
if 'admin_counter' not in ss:
        ss.admin_counter = 0

track_and_assign, create_,view_image = util.st.tabs(["Track & Assign", "Create","View_Image"])

with track_and_assign:
    # Assign the QC projects to the users 
    util.st.markdown("## **Assign QC**")
    # Multiselect box to select dataset
    ss['dn']=util.get_all_dataset()
    dataset = util.st.selectbox("Dataset", options=ss['dn'])
    ss.dataset = dataset
    # Multiselect box to select QC users
    QCusers = util.get_QC_users()
    assignTo = util.st.multiselect("QC User(s)", options=QCusers)
    if util.st.button("Assign", use_container_width=True):
        # The 'assign' button is clicked, and you can now access the selected options
        util.assign_dataset(dataset,assignTo)
    util.st.divider()
    util.st.markdown("## **Splitted Data**")
    data = []
    for i in ss['dn']:
        users = util.user_names(i)
        data.append({'dataset': i, 'users': ', '.join(users)})
    df = util.pd.DataFrame(data)
    util.st.dataframe(df, use_container_width=True, hide_index=True)
    util.st.divider()

    ###TRACK QC
    util.st.markdown("## **Track QC**")
    with util.psycopg2.connect(**util.connection_params) as conn,conn.cursor() as cur:
        query = "SELECT * FROM project"
        cur.execute(query)
        result_tuples = cur.fetchall()
        column_names = [desc[0] for desc in cur.description]
        df = util.pd.DataFrame(result_tuples, columns=column_names)
    util.st.dataframe(df, hide_index=True, use_container_width=True)
    util.st.button("Refresh",on_click=util.refresh_Dataset, use_container_width=True)
    util.st.divider()
 
    # Display entire dataset
    util.st.markdown("## **Display Dataframe**")
    select=util.st.selectbox("Show Dataset:",options=ss['dn'])
    button_show="show_dataframe"
    if select and util.st.button("Show",key=button_show, use_container_width=True):
        frame=util.display_dataset(select)
        csv = util.df_to_csv(frame)
        util.st.download_button(
            label="Download",
            data=csv,
            file_name=f'{select}.csv',
            mime='text/csv',use_container_width=True
        )
        util.st.write(frame)
    
    util.st.divider()

    #Display individual progress
    util.st.markdown("## **User Progress**")
    progress_dataset=util.st.selectbox("Show Dataset:",options=ss['dn'],key='progress_dataset')
    progress_button="Display"
    if progress_dataset:
        progress_users=util.user_names(progress_dataset)
        progress_user=util.st.selectbox("Show Users:",options=progress_users)
        util.st.button("Display",key=progress_button,use_container_width=True)
        if progress_user and util.st.button:
            id=util.get_dataset_id(progress_dataset)
            uid=util.get_person_id(progress_user)
            csv=util.get_user_dataframe(id,uid)
            util.st.write(csv)
    else:
        util.st.info("User Not Assigned any Dataset")
        
#Create Dataset
with create_:
    #Add QC_users
    util.st.markdown("## **Create User**")
    user=util.st.text_input("Username")
    password=util.st.text_input("Password")
    button_key = "create_user_button"
    if user and password and util.st.button("Create User", key=button_key, use_container_width=True, type="primary"):
        util.check_login(user, password)
    util.st.divider()

    #create dataset
    util.st.markdown("## **Create Dataset**")
    dataset=util.st.text_input("Dataset:")
    dataset_type_options = ["DOTA", "PASCAL_VOC"]
    dataset_type = util.st.selectbox("Dataset Type", options=dataset_type_options,index=None) 
    #Check the Valid Folders
    image_folder_path = util.st.text_input("Enter image folder path:", help="Define the folder location for images")
    if util.os.path.isdir(image_folder_path):
        util.st.success(f"Valid Image Folder!", icon="✅")
        valid_image_path = True
    elif image_folder_path == "":
        util.st.info('Enter directory of Images', icon="ℹ️")
        valid_image_path = False
    else:
        util.st.error("Please enter a valid image folder path", icon='❌')
        valid_image_path = False

    label_folder_path = util.st.text_input("Enter label folder path:", help="Define the folder location for labels")

    if util.os.path.isdir(label_folder_path):
        util.st.success(f"Valid Label Folder!", icon="✅")
        valid_label_path = True
    elif label_folder_path == "":
        util.st.info('Enter directory of Labels', icon="ℹ️")
        valid_label_path = False
    else:
        util.st.error("Please enter a valid label folder path", icon='❌')
        valid_label_path = False
    #QC_Percentage
    qc_per = util.st.number_input("Enter QC Percentage:", step=1)
    ss.qc_per=qc_per
    if dataset and valid_image_path and valid_label_path and util.st.button("Create") and ss.qc_per:
        util.create_dataset(dataset, image_folder_path, label_folder_path,ss.qc_per, dataset_type)
### Admin Section To View Whole Dataset With Images ###
with view_image:
    image_dataset=util.st.selectbox("Show Dataset:",options=ss['dn'],key='image_dataset',index=None)
    view_image_button_admin = util.st.button("View",key='view_image_button',use_container_width=True, type="primary", on_click=enable_view)
    if image_dataset and ss.view_image_status:
        view_image_folder,view_label_folder=util.get_paths(image_dataset)
        ss.dataset_name=image_dataset
        ss.image_folder_path = view_image_folder
        ss.label_folder_path = view_label_folder
        #dataset_type - To See Whether it is Like Dota or Pascal_Voc
        view_dataset_type=util.fetch_dataset_type(image_dataset)
        #Total DataFrame of Dataset
        ss.df =util.display_dataset(image_dataset)
        #Image List in it
        ss.image_list = ss.df['Image'].values
        ss.qc_image_list = ss.df['Image'].values
        ss.length_of_qc_list = len(ss.qc_image_list)
        ss.stat_image_list = [f for f in util.os.listdir(view_image_folder) if f.endswith(('.png', '.jpg'))]
        # Dataframe
        ss.img_quality_list = ss.df['Image Quality'].values.tolist()
        ss.remarks_list = ss.df['Remarks'].values.tolist()
        ss.tp = ss.df['True_Positive'].values.tolist()
        ss.fp = ss.df['False_Positive'].values.tolist()
        ss.tn = ss.df['True_Negative'].values.tolist()
        ss.fn = ss.df['False_Negative'].values.tolist()
        m1,m2=util.st.columns((7,3))

        # util.st.write(len(ss.df['Image']))

        with m1:
            with util.st.container(border=True):
                util.st.markdown(f"<p style='display: flex; justify-content: space-between;'><span style='color:green'>{image_dataset.upper()}</span><span style='color:blue'>{view_dataset_type}</span></p>", unsafe_allow_html=True)
                all_classes, _, _, _ = util.calculate_dataset_stats(ss.qc_image_list, view_label_folder, ss.dataset_name)
                ss.label_color_dict = util.generate_color(all_classes)
                #to get image and boolean value for negative and non negative image
                image, is_neg = util.annotate_image(util.os.path.join(view_image_folder,ss.df['Image'][ss.admin_counter]), view_label_folder, util.fetch_dataset_type(image_dataset),ss.label_color_dict, view_labels=True)
                toggle_col, neg_image_alert = util.st.columns([0.5, 0.5])
                with toggle_col:
                    view_img = util.st.button("View Image", on_click=toggle_view_image, use_container_width=True)
                with neg_image_alert:
                    if is_neg:
                        util.st.markdown(" <h3 style='text-align:right; color:maroon;'>Negative Image</h3>", unsafe_allow_html=True)


                if ss.view_image:
                    # Display image if "view_img" is on; this allows zooming & panning into the image
                    fig = util.sp.make_subplots(rows=1, cols=1)
                    fig.add_trace(util.go.Image(z=image))
                    fig.update_layout(height=690, margin={'b':0, 't':0}) # Remove the white space above and below the plots
                    # Hide axes information
                    fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
                    fig.update_yaxes(visible=False,showgrid=False,zeroline=False)
                    util.st.plotly_chart(fig, use_container_width=True)
                    util.st.caption(f"<center>{ss.qc_image_list[ss.admin_counter]}</center>", unsafe_allow_html=True)
                    hide_img_fs = '''
                                <style>
                                button[title="View fullscreen"]{
                                    visibility: hidden;}
                                </style>
                                '''

                    util.st.markdown(hide_img_fs, unsafe_allow_html=True)
                elif not ss.view_image:
                    util.st.image(image, use_column_width=True, caption=f"{ss.qc_image_list[ss.admin_counter]}")
                    hide_img_fs = '''
                                <style>
                                button[title="View fullscreen"]{
                                    visibility: hidden;}
                                </style>
                                '''

                    util.st.markdown(hide_img_fs, unsafe_allow_html=True)
            c1, c2, c3= util.st.columns(3)

            with c1:
                if ss.admin_counter > 0:
                    prevButton = util.st.button("Prev", use_container_width=True, key="previous", on_click=onPrev)

            with c2:
                if ss.admin_counter >= 0 and ss.admin_counter <= len(ss.qc_image_list) - 1:
                    # Replace the existing index section with an input box/slider for index
                    index = util.st.number_input(
                        f"{ss.admin_counter} / {len(ss.qc_image_list)}",
                        value=ss.admin_counter,
                        min_value=0,
                        max_value=len(ss.qc_image_list) - 1, 
                        label_visibility='collapsed',
                        key='admin_image_index',
                        on_change=update_index)
                    
        
            with c3:
                if ss.admin_counter < len(ss.qc_image_list) - 1:
                    nextButton = util.st.button("Next", use_container_width=True, key="next", on_click=onNext)
        img_quality = util.st.text_input(label="Image Quality",key='admin_img_quality', value =ss['df']['Image Quality'][ss.admin_counter] if not util.pd.isnull(ss['df']['Image Quality'][ss.admin_counter]) else "", disabled=True)
        remarks = util.st.text_input("Remarks", key='admin_remarks', value=ss.df['Remarks'][ss.admin_counter] if not util.pd.isnull(ss.df['Remarks'][ss.admin_counter]) else "", disabled=True)
        res=util.trackQCprogress(ss)
        util.st.text_input("Total QC'ed", value=f'{res[0]}/{len(ss.qc_image_list)}', disabled=True) 
        util.st.text_input("Classification accuracy",value=f'{res[1]}', disabled=True)
        util.st.text_input("Recall",value=f'{res[2]}', disabled=True)
        util.st.text_input("Precision",value=f'{res[3]}', disabled=True)
        util.st.text_input("f1_score",value=f'{res[4]}', disabled=True)
        with m2:
            t_p = util.st.text_input("True_Positive", key='admin_tp', value=ss.df['True_Positive'][ss.admin_counter] if not util.pd.isnull(ss.df['True_Positive'][ss.admin_counter]) else "", disabled=True)

            f_p = util.st.text_input("False_Positive", key='admin_fp', value=ss.df['False_Positive'][ss.admin_counter] if not util.pd.isnull(ss.df['False_Positive'][ss.admin_counter]) else "", disabled=True)

            no_obj_ann = util.st.text_input("No_of_Objects_Annotated", key="admin_no_obj_annotated", value=util.single_image_ann_info(ss.qc_image_list[ss.admin_counter], view_label_folder, util.fetch_dataset_type(image_dataset)), disabled=True)

            t_n = util.st.text_input("True_Negative", key='admin_tn', value=ss.df['True_Negative'][ss.admin_counter] if not util.pd.isnull(ss.df['True_Negative'][ss.admin_counter]) else "", disabled=True)
            
            f_n = util.st.text_input("False_Negative", key='admin_fn', value=ss.df['False_Negative'][ss.admin_counter] if not util.pd.isnull(ss.df['False_Negative'][ss.admin_counter]) else "", disabled=True)
        if ss.view_image_status:
            util.fetch_dataset_stats()
