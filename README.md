# Live Face Recognition App
This is a Streamlit App that allows you to recognize faces in a live webcam video stream. Check out the app on the [Streamlit Community Cloud](https://streamlit.io/cloud) by clicking on the badge below:

[![Streamlit Community Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://livefaceid.streamlit.app)

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Last Commit](https://img.shields.io/github/last-commit/martlgap/livefaceidapp)](https://img.shields.io/github/last-commit/martlgap/livefaceidapp)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# 👀 Example
As an example, instead of using a webcam I fed a snippet of BigBangTheory into the app and it recognized the characters in real-time:


![Demo](example.gif)


# 🚀 Usage
## Run the App on Streamlit Community Cloud
You can run my app in your Browser using the [Streamlit Community Cloud](https://streamlit.io/cloud) by clicking on the badge below:

[![Open in Streamlit Community](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://livefaceid.streamlit.app)


## Locally on your machine (Tested on MacOS, Linux)
You can run the app locally on your machine by cloning this repository and running the following commands:

```bash
git clone https://github.com/Martlgap/livefaceidapp.git
cd livefaceidapp
pip -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

Depending on platform and hardware you need to install onnxruntime, onnxruntime-gpu, or onnxruntime-silicon.

## Locally on server (Tested on Linux)
If you want to run the app on a server in your local network and would like to access it from another machine in the same network, you can use the [ssl-proxy](https://github.com/suyashkumar/ssl-proxy) plugin of [suyashkumar](https://github.com/suyashkumar):

```bash
wget https://github.com/suyashkumar/ssl-proxy/releases/download/v0.2.7/ssl-proxy-linux-amd64.tar.gz

gzip -d ssl-proxy-linux-amd64.tar.gz
tar -xvf ssl-proxy-linux-amd64.tar

./ssl-proxy-linux-amd64 -from 0.0.0.0:8502 -to 0.0.0.0:8501
```
After that you are able to access the app in your browser via https://your-server-ip:8502


# 🖥️ Streaming Servers
For beeing able to use the app on the Streamlit Communnity Cloud, I set up a TWILIO TURN server with my credentials. Those are saved in the streamlit community cloud secrets, which is not included in this repository. If you want to use the app on your own server, you have to set up your own TURN server. You can find a detailed description on how to do that [here](https://www.twilio.com/docs/stun-turn).

You can set your own credentials by exporting them as environment variables:
```bash
export TWILIO_ACCOUNT_SID=your-twilio-account-sid
export TWILIO_AUTH_TOKEN=your-twilio-auth-token
```


# ⚙️ How it Works
A detailed description of the implementation can be found here: 

[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://martlgap.medium.com/529fc686b475)


# 🧠 Machine Learning Models
The app uses the following machine learning models:
- Face Detection: FaceMesh ([MediaPipe](https://google.github.io/mediapipe/solutions/face_mesh.html))
- Face Recognition: [MobileNetV2](https://arxiv.org/abs/1801.04381) Architecture, trained with [MS1M](https://arxiv.org/abs/1607.08221) dataset and [ArcFace](https://arxiv.org/abs/1801.07698) Loss using [Tensorflow](https://tensorflow.org), converted to [ONNX](https://onnxruntime.ai)


# 📖 About
I developed this app during a project at the [Chair of Human-Machine Communication](https://www.ce.cit.tum.de/en/mmk/home/). The goal was to compare live face recognition systems across different platforms. 


# 📚 Resources
Please take a look at the following resources, which helped me a lot during the development of this app:
- [Streamlit](https://streamlit.io/)
- [Streamlit Community](https://discuss.streamlit.io/)
- [Streamlit Webrtc](https://github.com/whitphx/streamlit-webrtc)
- [ssl-proxy](https://github.com/suyashkumar/ssl-proxy)
- [MediaPipe](https://mediapipe-studio.webapps.google.com/home)
- [aiortc](https://github.com/aiortc/aiortc)
- [WebRTC](https://webrtc.org/)
- [WebRTC Samples](https://webrtc.github.io/samples/)
- [Deployment](https://www.artefact.com/blog/how-to-deploy-and-secure-your-streamlit-app-on-gcp/)
- [Real-Time-Video-Streams](https://betterprogramming.pub/real-time-video-streams-with-streamlit-webrtc-bd38d15f2ef3)


# 🪲 BUGS - KNOWN ISSUES - TODOS:
- [ ] Running the app on streamlit community cloud introduces a severe lag/delay in the video stream. This is also present when working with the app locally on a server. There should be a way to reduce the delay, either by setting the buffer size smaller or by using frame dropping.
- [ ] There is also sometimes a lag when running the app locally on your machine. This is due to several reasons and need to be investigated further.
- [ ] Aspect ratio bug on smartphone
