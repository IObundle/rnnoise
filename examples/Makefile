
input_to_pcm:
	ffmpeg -i Recording.wav -f s16le -acodec pcm_s16le file.pcm

output_to_wav: rnn_noise
	ffmpeg -f s16le -ac 1 -ar 48000 -i out.pcm out.wav

rnn_noise: input_to_pcm
	./rnnoise_demo file.pcm out.pcm

clean:
	rm file.pcm out.pcm out.wav
