input_to_pcm:
	ffmpeg -i Recording.wav -f s16le -acodec pcm_s16le pcm.in

output_to_wav: rnn_noise
	ffmpeg -f s16le -ac 1 -ar 48000 -i pcm.out out.wav

rnn_noise: input_to_pcm
	./rnnoise_demo pcm.in pcm.out

clean:
	rm pcm.in pcm.out out.wav Recording.wav
