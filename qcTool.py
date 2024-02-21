# Libraries
import pages.utilities as util

util.clear_cache()
ss = util.st.session_state
################################
#######   PAGE CONFIG ##########
################################
# util.clear()

util.st.set_page_config(page_title="Login ",
                   page_icon=":game_die:",
                   initial_sidebar_state="collapsed")

# util.st.sidebar.title("Pages")
util.show_pages_from_config()
util.hide_pages(["QC Information", "QC Tool","Admin Dashboard"])

################################
#######      UI      ###########
################################

util.st.markdown("# <center> Annotation QC Tool! 🚀 </center> ", unsafe_allow_html=True)
util.st.divider()
if "startQC" in ss:
    ss.startQC = None
if "valid_user" not in ss:
    ss.valid_user = None
#Login Check #
def user_login():
    username_ = util.st.text_input("Username", key="inputUsername")
    password_ = util.st.text_input("Password", type="password", key="inputPassword")
    
    if username_ == "" or password_ == "":
        util.st.info("Username/Password field blank. Please enter credentials.", icon="ℹ️")
        ss.valid_user = False
        return False

    if util.read_user_credentials(username_,password_):
        util.st.success("Login successful!", icon="✅")
        ss.valid_user = True
        ss.username = username_
        ss.password = password_
        return True

    else:
        util.st.error("Invalid credentials. Please try again.", icon='❌')
        ss.valid_user = False
        return False
#Admin Check - User Check
if user_login():
    if ss.username != "admin" and ss.valid_user != None:
        util.st.switch_page("pages/qcInfo.py")
    else:
        util.st.switch_page("pages/admin.py")
else:
    util.st.stop()
