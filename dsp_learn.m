x = randn(1,100);
w = 10;
y = conv(ones(1,w)/w, x);
avgs = y(10:99);
plot(avgs)
hold on
plot(x);
legend('--y', '--x')