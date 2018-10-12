rocks = ['CG', 'KM', 'LG', 'GAR', 'KE', 'NE', 'GAW', 'KW', 'MA'];
actual = [0.21, 0.25, 0.19, 0.25, 0.32, 0.28, 0.22, 0.29, 0.31];
predicted = [0.23, 0.249, 0.18, 0.23, 0.31, 0.282, 0.23, 0.296, 0.32];

index = 1:9;

plot(index, actual)
hold on
plot(index, predicted)
ylim([0 0.36])
legend('Actual', 'Predicted')
xlabel('Rock Sample')
ylabel('DRI')

