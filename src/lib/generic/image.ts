export type Rect = {
	x: number;
	y: number;
	width: number;
	height: number;
};

/**
 * Draws src onto dest, fiilling it without stretching the image
 *
 * @param dest destination canvas
 * @param destBackgroundColor used to fill the empty bands either on the sides, or at the top and bottom
 * @param src the source image
 * @param srcRect an optional rectangle indicating which part of the source image we want.
 */
export function drawImageFitted(
	dest: CanvasRenderingContext2D,
	destBackgroundColor: string,
	src: HTMLImageElement | HTMLCanvasElement,
	srcRect?: Rect
) {
	if (!srcRect) {
		srcRect = { x: 0, y: 0, width: src.width, height: src.height };
	}
	const destCanvas = dest.canvas;

	let dWidth = destCanvas.width;
	let dHeight = destCanvas.height;

	const widthRatio = destCanvas.width / srcRect.width;
	const heightRatio = destCanvas.height / srcRect.height;
	if (widthRatio > heightRatio) {
		// We chop off the left and right
		dWidth = destCanvas.width * (srcRect.width / srcRect.height);
	} else {
		// We chop off the top and bottom
		dHeight = destCanvas.height * (srcRect.height / srcRect.width);
	}

	const dx = (destCanvas.width - dWidth) / 2;
	const dy = (destCanvas.height - dHeight) / 2;
	dest.fillStyle = destBackgroundColor;
	dest.fillRect(0, 0, destCanvas.width, destCanvas.height);
	dest.drawImage(src, srcRect.x, srcRect.y, srcRect.width, srcRect.height, dx, dy, dWidth, dHeight);
}

/**
 * @param image an inverted grayscale image
 * @param threshold  the luminance value at which we consider there is a pixel there (and not the background)
 * @returns
 */
export function findBoundingBox(image: ImageData, threshold: number): Rect {
	let top = Number.MAX_SAFE_INTEGER;
	let bottom = Number.MIN_SAFE_INTEGER;
	let left = Number.MAX_SAFE_INTEGER;
	let right = Number.MIN_SAFE_INTEGER;

	let foundAPixel = false;

	const { data } = image;
	for (let i = 0; i < data.length; i += 4) {
		const x = (i / 4) % image.width;
		const y = (i / 4 / image.width) | 0;
		if (data[i] > threshold) {
			foundAPixel = true;
			if (y < top) top = y;
			if (x < left) left = x;
			if (y > bottom) bottom = y;
			if (x > right) right = x;
		}
	}

	if (foundAPixel) {
		return { x: left, y: top, width: right - left, height: bottom - top };
	} else {
		return { x: 0, y: 0, width: image.width, height: image.height };
	}
}
/**
 * @param px bytes representing an image in rgba format.
 * It gets modified in place.
 * The original alpha is preserved
 */
export function blueToGrayscaleInverted(px: Uint8ClampedArray) {
	for (let i = 0; i < px.length; i += 4) {
		const r = px[i];
		const g = px[i + 1];
		const b = px[i + 2];
		const luminance = r == 0 && g == 0 ? b : 0;
		px[i] = luminance;
		px[i + 1] = luminance;
		px[i + 2] = luminance;
	}
}

/**
 * Luminance computation from:
 * https://www.dynamsoft.com/blog/insights/image-processing/image-processing-101-color-space-conversion/
 *
 * @param px bytes representing an image in rgba format.
 * It gets modified in place.
 * The original alpha is preserved
 */
export function toGrayscaleInverted(px: Uint8ClampedArray): void {
	for (let i = 0; i < px.length; i += 4) {
		const r = px[i];
		const g = px[i + 1];
		const b = px[i + 2];
		const luminance = 255 - Math.floor(0.299 * r + 0.587 * g + 0.114 * b);
		px[i] = luminance;
		px[i + 1] = luminance;
		px[i + 2] = luminance;
	}
}

/**
 * Converts an HSV color value to RGB. Conversion formula
 * Assumes h, s, and v are contained in the set [0, 1] and
 * returns r, g, and b in the set [0, 255].
 *
 * From:  https://stackoverflow.com/questions/17242144/javascript-convert-hsb-hsv-color-to-rgb-accurately/
 */
export function hsv2rgb(h: number, s: number, v: number): number[] {
	const i = Math.floor(h * 6);
	const f = h * 6 - i;
	const p = v * (1 - s);
	const q = v * (1 - f * s);
	const t = v * (1 - (1 - f) * s);
	let r = 0;
	let g = 0;
	let b = 0;
	switch (i % 6) {
		case 0:
			((r = v), (g = t), (b = p));
			break;
		case 1:
			((r = q), (g = v), (b = p));
			break;
		case 2:
			((r = p), (g = v), (b = t));
			break;
		case 3:
			((r = p), (g = q), (b = v));
			break;
		case 4:
			((r = t), (g = p), (b = v));
			break;
		case 5:
			((r = v), (g = p), (b = q));
			break;
	}
	return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}
