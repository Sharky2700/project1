import sys
import argparse
import jetson.utils
import jetson.inference


parser = argparse.ArgumentParser()
parser.add_argument("--input_location", type=str, default="/dev/video0", help="Location of the camera/video file(The default is the USB camera)")
parser.add_argument("--output_location", type=str, default="display://0", help="Location of the output(my_video.mp4 for a file output)")



try:
	args=parser.parse_args()
except:
	print("")
	parser.print_help()
	sys.exit(0)

net = jetson.inference.detectNet(argv=["--model=basketball_model.onnx", "--labels=labels.txt", "--input_blob=input_0", "--output-cvg=scores", "--output-bbox=boxes"], threshold=0.3)
camera  = jetson.utils.videoSource(args.input_location)
display = jetson.utils.videoOutput(args.output_location)

while display.IsStreaming():
	img = camera.Capture()
	detections = net.Detect(img)
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

