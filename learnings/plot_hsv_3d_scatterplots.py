def plot_hsv_3d_scatterplot(self):
    img = self.image.copy()

    pixel_colors = img.reshape((np.shape(img)[0] * np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_img)
    fig = plt.figure(figsize=(20, 10))
    axis = fig.add_subplot(1, 2, 1, projection="3d")
    axis.scatter(h.flatten(), v.flatten(), s.flatten(),
                 facecolors=pixel_colors,
                 marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Value")
    axis.set_zlabel("Saturation")

    axis = fig.add_subplot(1, 2, 2, projection="3d")
    axis.scatter(s.flatten(), h.flatten(), v.flatten(),
                 facecolors=pixel_colors,
                 marker=".")
    axis.set_xlabel("Saturation")
    axis.set_ylabel("Hue")
    axis.set_zlabel("Value")

    plt.show()