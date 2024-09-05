import ffmpeg

input_file = ffmpeg.input('assets/city_camera.ts')

output_file = ffmpeg.output(input_file, 'assets/city_camera.mp4')
ffmpeg.run(output_file)