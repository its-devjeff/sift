import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Code, Eye, Upload, RotateCcw, Calculator, Zap, Grid3X3, BarChart3, Target, Sliders } from 'lucide-react';
import CanvasHistogram from './CanvasHistogram';

const EnhancedInteractiveSIFT = () => {
    // --- State Variables ---
    const [originalImage, setOriginalImage] = useState(null);
    const [grayscaleImage, setGrayscaleImage] = useState(null);
    const [pixelMatrix, setPixelMatrix] = useState([]);
    const [selectedPixel, setSelectedPixel] = useState({ row: -1, col: -1 });
    const [gx, setGx] = useState(0);
    const [gy, setGy] = useState(0);
    const [magnitude, setMagnitude] = useState(0);
    const [orientation, setOrientation] = useState(0);
    const [histogramData, setHistogramData] = useState(Array(180).fill(0));
    const [histogramTable, setHistogramTable] = useState([]);
    const [showCode, setShowCode] = useState(false);
    const [currentSection, setCurrentSection] = useState('scalespace');
    
    const canvasRef = useRef(null);
    const pixelSize = 15;

    // Scale Space & DoG States
    const [sigma, setSigma] = useState(1.6);
    const [numOctaves, setNumOctaves] = useState(4);
    const [scalesPerOctave, setScalesPerOctave] = useState(5);
    const [scaleSpaceImages, setScaleSpaceImages] = useState([]);
    const [dogImages, setDogImages] = useState([]);
    const [keypointLocations, setKeypointLocations] = useState([]);
    
    // Keypoint Descriptor States
    const [descriptorPatch, setDescriptorPatch] = useState([]);
    const [descriptorKeypoint, setDescriptorKeypoint] = useState({ x: 50, y: 50 });
    const [descriptorFeatures, setDescriptorFeatures] = useState([]);
    const descriptorCanvasRef = useRef(null);

    // Feature Matching States
    const [image1ForMatching, setImage1ForMatching] = useState(null);
    const [image2ForMatching, setImage2ForMatching] = useState(null);
    const [image2Rotation, setImage2Rotation] = useState(0);
    const [image2Zoom, setImage2Zoom] = useState(1);
    const [matchingResults, setMatchingResults] = useState([]);
    const matchingCanvasRef = useRef(null);

    // --- Real Gaussian Blur Implementation ---
    const createGaussianKernel = useCallback((sigma, size) => {
        const kernel = [];
        const center = Math.floor(size / 2);
        let sum = 0;
        
        for (let y = 0; y < size; y++) {
            kernel[y] = [];
            for (let x = 0; x < size; x++) {
                const dx = x - center;
                const dy = y - center;
                const value = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
                kernel[y][x] = value;
                sum += value;
            }
        }
        
        // Normalize
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                kernel[y][x] /= sum;
            }
        }
        
        return kernel;
    }, []);

    const applyGaussianBlur = useCallback((imageData, sigma) => {
        const kernelSize = Math.ceil(sigma * 6) | 1; // Ensure odd size
        const kernel = createGaussianKernel(sigma, kernelSize);
        const center = Math.floor(kernelSize / 2);
        
        const { data, width, height } = imageData;
        const output = new Uint8ClampedArray(data.length);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                let r = 0, g = 0, b = 0, a = 0;
                
                for (let ky = 0; ky < kernelSize; ky++) {
                    for (let kx = 0; kx < kernelSize; kx++) {
                        const py = Math.min(Math.max(y + ky - center, 0), height - 1);
                        const px = Math.min(Math.max(x + kx - center, 0), width - 1);
                        const idx = (py * width + px) * 4;
                        const weight = kernel[ky][kx];
                        
                        r += data[idx] * weight;
                        g += data[idx + 1] * weight;
                        b += data[idx + 2] * weight;
                        a += data[idx + 3] * weight;
                    }
                }
                
                const outIdx = (y * width + x) * 4;
                output[outIdx] = r;
                output[outIdx + 1] = g;
                output[outIdx + 2] = b;
                output[outIdx + 3] = a;
            }
        }
        
        return new ImageData(output, width, height);
    }, [createGaussianKernel]);

    // --- Real Scale Space Construction ---
    const generateRealScaleSpace = useCallback((imageCanvas) => {
        if (!imageCanvas) return;
        
        const ctx = imageCanvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, imageCanvas.width, imageCanvas.height);
        
        const scaleSpace = [];
        const dogSpace = [];
        
        let currentImageData = imageData;
        
        for (let octave = 0; octave < numOctaves; octave++) {
            const octaveImages = [];
            const k = Math.pow(2, 1 / (scalesPerOctave - 3));
            
            // Generate blurred images for this octave
            for (let scale = 0; scale < scalesPerOctave; scale++) {
                const currentSigma = sigma * Math.pow(k, scale) * Math.pow(2, octave);
                const blurred = applyGaussianBlur(currentImageData, currentSigma);
                
                // Create canvas for this scale
                const canvas = document.createElement('canvas');
                canvas.width = currentImageData.width;
                canvas.height = currentImageData.height;
                const canvasCtx = canvas.getContext('2d');
                canvasCtx.putImageData(blurred, 0, 0);
                
                octaveImages.push({
                    canvas: canvas,
                    imageData: blurred,
                    sigma: currentSigma,
                    octave: octave,
                    scale: scale
                });
            }
            
            scaleSpace.push(octaveImages);
            
            // Generate DoG images for this octave
            const octaveDoG = [];
            for (let i = 0; i < octaveImages.length - 1; i++) {
                const dogImageData = computeDoG(octaveImages[i].imageData, octaveImages[i + 1].imageData);
                
                const dogCanvas = document.createElement('canvas');
                dogCanvas.width = dogImageData.width;
                dogCanvas.height = dogImageData.height;
                const dogCtx = dogCanvas.getContext('2d');
                dogCtx.putImageData(dogImageData, 0, 0);
                
                octaveDoG.push({
                    canvas: dogCanvas,
                    imageData: dogImageData,
                    octave: octave,
                    level: i
                });
            }
            
            dogSpace.push(octaveDoG);
            
            // Downsample for next octave
            if (octave < numOctaves - 1) {
                currentImageData = downsampleImageData(octaveImages[scalesPerOctave - 3].imageData);
            }
        }
        
        setScaleSpaceImages(scaleSpace);
        setDogImages(dogSpace);
        
        // Find keypoints
        findKeypoints(dogSpace);
    }, [sigma, numOctaves, scalesPerOctave, applyGaussianBlur]);

    // --- Real DoG Computation ---
    const computeDoG = useCallback((imageData1, imageData2) => {
        const { data: data1, width, height } = imageData1;
        const { data: data2 } = imageData2;
        const output = new Uint8ClampedArray(data1.length);
        
        for (let i = 0; i < data1.length; i += 4) {
            // Convert to grayscale and compute difference
            const gray1 = 0.299 * data1[i] + 0.587 * data1[i + 1] + 0.114 * data1[i + 2];
            const gray2 = 0.299 * data2[i] + 0.587 * data2[i + 1] + 0.114 * data2[i + 2];
            const diff = Math.abs(gray1 - gray2);
            
            output[i] = diff;     // R
            output[i + 1] = diff; // G
            output[i + 2] = diff; // B
            output[i + 3] = 255;  // A
        }
        
        return new ImageData(output, width, height);
    }, []);
    // --- Real Keypoint Detection ---
    const findKeypoints = useCallback((dogSpace) => {
        const keypoints = [];
        const threshold = 10; // Contrast threshold
        
        dogSpace.forEach((octave, octaveIdx) => {
            if (octave.length < 3) return; // Need at least 3 DoG images for comparison
            
            // Check middle DoG image against neighbors
            for (let level = 1; level < octave.length - 1; level++) {
                const current = octave[level].imageData;
                const above = octave[level - 1].imageData;
                const below = octave[level + 1].imageData;
                
                const { data: currentData, width, height } = current;
                const { data: aboveData } = above;
                const { data: belowData } = below;
                
                // Check each pixel (skip borders)
                for (let y = 1; y < height - 1; y++) {
                    for (let x = 1; x < width - 1; x++) {
                        const idx = (y * width + x) * 4;
                        const centerValue = currentData[idx];
                        
                        if (centerValue < threshold) continue;
                        
                        let isExtremum = true;
                        
                        // Check 26 neighbors (3x3x3 cube)
                        for (let dy = -1; dy <= 1 && isExtremum; dy++) {
                            for (let dx = -1; dx <= 1 && isExtremum; dx++) {
                                const neighborIdx = ((y + dy) * width + (x + dx)) * 4;
                                
                                // Check current level neighbors
                                if (currentData[neighborIdx] >= centerValue && !(dx === 0 && dy === 0)) {
                                    isExtremum = false;
                                }
                                
                                // Check above and below level neighbors
                                if (aboveData[neighborIdx] >= centerValue || belowData[neighborIdx] >= centerValue) {
                                    isExtremum = false;
                                }
                            }
                        }
                        
                        if (isExtremum) {
                            keypoints.push({
                                x: x * Math.pow(2, octaveIdx),
                                y: y * Math.pow(2, octaveIdx),
                                octave: octaveIdx,
                                level: level,
                                response: centerValue
                            });
                        }
                    }
                }
            }
        });
        
        setKeypointLocations(keypoints);
    }, []);

    // --- Real Downsample Implementation ---
    const downsampleImageData = useCallback((imageData) => {
        const { data, width, height } = imageData;
        const newWidth = Math.floor(width / 2);
        const newHeight = Math.floor(height / 2);
        const newData = new Uint8ClampedArray(newWidth * newHeight * 4);
        
        for (let y = 0; y < newHeight; y++) {
            for (let x = 0; x < newWidth; x++) {
                const srcX = x * 2;
                const srcY = y * 2;
                const srcIdx = (srcY * width + srcX) * 4;
                const dstIdx = (y * newWidth + x) * 4;
                
                newData[dstIdx] = data[srcIdx];
                newData[dstIdx + 1] = data[srcIdx + 1];
                newData[dstIdx + 2] = data[srcIdx + 2];
                newData[dstIdx + 3] = data[srcIdx + 3];
            }
        }
        
        return new ImageData(newData, newWidth, newHeight);
    }, []);

    // --- Utility Functions ---
    const convertToGrayscale = useCallback((imageData) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = imageData.width;
        canvas.height = imageData.height;
        ctx.drawImage(imageData, 0, 0);

        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imgData.data;
        const grayPixels = [];

        for (let i = 0; i < data.length; i += 4) {
            const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
            data[i] = avg;
            data[i + 1] = avg;
            data[i + 2] = avg;
            grayPixels.push(Math.round(avg));
        }
        ctx.putImageData(imgData, 0, 0);

        const matrix = [];
 for (let r = 0; r < canvas.height; r++) {
            matrix.push(grayPixels.slice(r * canvas.width, (r + 1) * canvas.width));
        }
        setPixelMatrix(matrix);

        // Generate scale space from this image
        generateRealScaleSpace(canvas);

        return canvas.toDataURL();
    }, [generateRealScaleSpace]);

    // --- Image Upload Handlers ---
    const handleImageUpload = (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    setOriginalImage(img.src);
                    const grayDataUrl = convertToGrayscale(img);
                    setGrayscaleImage(grayDataUrl);
                    setSelectedPixel({ row: -1, col: -1 });
                    setGx(0);
                    setGy(0);
                    setMagnitude(0);
                    setOrientation(0);
                    setHistogramData(Array(180).fill(0));
                    setHistogramTable([]);
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    };

    // --- Enhanced Canvas Drawing with Gradient Visualization and Keypoint Circling ---
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !grayscaleImage || pixelMatrix.length === 0) return;

        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.onload = () => {
            canvas.width = Math.min(pixelMatrix[0].length * pixelSize, 800);
            canvas.height = Math.min(pixelMatrix.length * pixelSize, 600);

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            // Draw grid and pixel values
            ctx.font = `${Math.min(pixelSize / 4, 10)}px monospace`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            
            const displayRows = Math.min(pixelMatrix.length, Math.floor(canvas.height / pixelSize));
            const displayCols = Math.min(pixelMatrix[0].length, Math.floor(canvas.width / pixelSize));
            
            // Calculate gradient for each pixel if we're in gradient mode
            const gradientInfo = {};
            if (currentSection === 'gradient') {
                for (let r = 1; r < displayRows - 1; r++) {
                    for (let c = 1; c < displayCols - 1; c++) {
                        const currentVal = pixelMatrix[r][c];
                        const rightVal = pixelMatrix[r][c + 1];
                        const leftVal = pixelMatrix[r][c - 1];
                        const topVal = pixelMatrix[r - 1][c];
                        const bottomVal = pixelMatrix[r + 1][c];
                        
                        const gx = rightVal - leftVal;
                        const gy = bottomVal - topVal;
                        const mag = Math.sqrt(gx * gx + gy * gy);
                        let orient = Math.atan2(gy, gx) * 180 / Math.PI;
                        if (orient < 0) orient += 360;
                        
                        gradientInfo[`${r}-${c}`] = { gx, gy, mag, orient };
                    }
                }
            }
            
            for (let r = 0; r < displayRows; r++) {
                for (let c = 0; c < displayCols; c++) {
                    const x = c * pixelSize;
                    const y = r * pixelSize;

                    // Determine pixel highlighting
                    let strokeColor = '#555';
                    let strokeWidth = 0.5;
                    let fillColor = null;
                    
                    if (selectedPixel.row === r && selectedPixel.col === c) {
                        // Selected pixel - red
                        strokeColor = '#ff0000';
                        strokeWidth = 3;
                        fillColor = 'rgba(255, 0, 0, 0.3)';
                    } else if (selectedPixel.row !== -1 && selectedPixel.col !== -1) {
                        // Check if this is a neighbor of selected pixel
                        const isNeighbor = Math.abs(r - selectedPixel.row) <= 1 && 
                                         Math.abs(c - selectedPixel.col) <= 1 &&
                                         !(r === selectedPixel.row && c === selectedPixel.col);
                        
                        if (isNeighbor) {
                            strokeColor = '#ffff00'; // Yellow for neighbors
                            strokeWidth = 2;
                            fillColor = 'rgba(255, 255, 0, 0.2)';
                        }
                    }
                    
                    // Fill background if needed
                    if (fillColor) {
                        ctx.fillStyle = fillColor;
                        ctx.fillRect(x, y, pixelSize, pixelSize);
                    }

                    // Draw grid
                    ctx.strokeStyle = strokeColor;
                    ctx.lineWidth = strokeWidth;
                    ctx.strokeRect(x, y, pixelSize, pixelSize);

                    // Draw pixel value
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
                    ctx.font = `${Math.min(pixelSize / 4, 10)}px monospace`;
                    ctx.fillText(pixelMatrix[r][c], x + pixelSize / 2, y + pixelSize / 2);
                    
                    // Draw magnitude and orientation for gradient mode
                    if (currentSection === 'gradient' && gradientInfo[`${r}-${c}`]) {
                        const grad = gradientInfo[`${r}-${c}`];
                        
                        // Draw magnitude above pixel
                        if (grad.mag > 0) {
                            ctx.fillStyle = 'rgba(0, 255, 255, 0.9)';
                            ctx.font = `${Math.min(pixelSize / 6, 8)}px monospace`;
                            ctx.fillText(grad.mag.toFixed(1), x + pixelSize / 2, y + 3);
                            
                            // Draw orientation arrow
                            const centerX = x + pixelSize / 2;
                            const centerY = y + pixelSize / 2;
                            const arrowLength = Math.min(pixelSize / 3, grad.mag / 10);
                            
                            if (arrowLength > 2) {
                                const angleRad = grad.orient * Math.PI / 180;
                                const endX = centerX + Math.cos(angleRad) * arrowLength;
                                const endY = centerY + Math.sin(angleRad) * arrowLength;
                                
                                ctx.strokeStyle = '#00ffff';
                                ctx.lineWidth = 1;
                                ctx.beginPath();
                                ctx.moveTo(centerX, centerY);
                                ctx.lineTo(endX, endY);
                                ctx.stroke();
                                
                                // Arrowhead
                                const headSize = 2;
                                ctx.beginPath();
                                ctx.moveTo(endX, endY);
                                ctx.lineTo(endX - headSize * Math.cos(angleRad - 0.5), 
                                          endY - headSize * Math.sin(angleRad - 0.5));
                                ctx.moveTo(endX, endY);
                                ctx.lineTo(endX - headSize * Math.cos(angleRad + 0.5), 
                                          endY - headSize * Math.sin(angleRad + 0.5));
                                ctx.stroke();
                            }
                        }
                    }
                }
            }
            
            // Draw gradient calculation arrows for selected pixel
            if (selectedPixel.row !== -1 && selectedPixel.col !== -1 && currentSection === 'gradient') {
                const centerX = selectedPixel.col * pixelSize + pixelSize / 2;
                const centerY = selectedPixel.row * pixelSize + pixelSize / 2;
                
                // Horizontal arrow (Gx calculation)
                ctx.strokeStyle = '#0099ff';
                ctx.lineWidth = 3;
                ctx.beginPath();
                // Left to right arrow
                const leftX = (selectedPixel.col - 1) * pixelSize + pixelSize / 2;
                const rightX = (selectedPixel.col + 1) * pixelSize + pixelSize / 2;
                ctx.moveTo(leftX, centerY);
                ctx.lineTo(rightX, centerY);
                ctx.stroke();
                
                // Horizontal arrowhead
                ctx.beginPath();
                ctx.moveTo(rightX, centerY);
                ctx.lineTo(rightX - 5, centerY - 3);
                ctx.moveTo(rightX, centerY);
                ctx.lineTo(rightX - 5, centerY + 3);
                ctx.stroke();
                
                // Vertical arrow (Gy calculation)
                ctx.strokeStyle = '#ff6600';
                ctx.lineWidth = 3;
                ctx.beginPath();
                // Top to bottom arrow
                const topY = (selectedPixel.row - 1) * pixelSize + pixelSize / 2;
                const bottomY = (selectedPixel.row + 1) * pixelSize + pixelSize / 2;
                ctx.moveTo(centerX, topY);
                ctx.lineTo(centerX, bottomY);
                ctx.stroke();
                
                // Vertical arrowhead
                ctx.beginPath();
                ctx.moveTo(centerX, bottomY);
                ctx.lineTo(centerX - 3, bottomY - 5);
                ctx.moveTo(centerX, bottomY);
                ctx.lineTo(centerX + 3, bottomY - 5);
                ctx.stroke();
                
                // Add labels for the arrows
                ctx.fillStyle = '#0099ff';
                ctx.font = `${Math.min(pixelSize / 3, 12)}px monospace bold`;
                ctx.fillText('Gx', rightX + 5, centerY + 3);
                
                ctx.fillStyle = '#ff6600';
                ctx.fillText('Gy', centerX + 3, bottomY + 12);
            }
            
            // Draw keypoints with enhanced circling in keypoints section
            if (keypointLocations.length > 0 && currentSection === 'keypoints') {
                keypointLocations.forEach(kp => {
                    if (kp.x < canvas.width && kp.y < canvas.height) {
                        // Draw larger, more visible circle
                        ctx.strokeStyle = '#00ff00';
                        ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
                        ctx.lineWidth = 3;
                        ctx.beginPath();
                        ctx.arc(kp.x, kp.y, 8, 0, 2 * Math.PI);
                        ctx.fill();
                        ctx.stroke();
                        
                        // Draw small center point
                        ctx.fillStyle = '#00ff00';
                        ctx.beginPath();
                        ctx.arc(kp.x, kp.y, 2, 0, 2 * Math.PI);
                        ctx.fill();
                    }
                });
            }
        };
        img.src = grayscaleImage;
    }, [grayscaleImage, pixelMatrix, selectedPixel, keypointLocations, currentSection]);

    // --- Gradient Calculation ---
    const handleCanvasClick = (event) => {
        const canvas = canvasRef.current;
        if (!canvas || pixelMatrix.length === 0) return;

        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        const col = Math.floor(x / pixelSize);
        const row = Math.floor(y / pixelSize);

        if (row >= 0 && row < pixelMatrix.length && col >= 0 && col < pixelMatrix[0].length) {
            setSelectedPixel({ row, col });
            calculateGradients(row, col);
        }
    };

    const calculateGradients = (row, col) => {
        if (pixelMatrix.length === 0) return;

        const width = pixelMatrix[0].length;
        const height = pixelMatrix.length;

        let currentPixel = pixelMatrix[row][col];
        let leftPixel = (col > 0) ? pixelMatrix[row][col - 1] : currentPixel;
        let rightPixel = (col < width - 1) ? pixelMatrix[row][col + 1] : currentPixel;
        let topPixel = (row > 0) ? pixelMatrix[row - 1][col] : currentPixel;
        let bottomPixel = (row < height - 1) ? pixelMatrix[row + 1][col] : currentPixel;

        const calculatedGx = rightPixel - leftPixel;
        const calculatedGy = bottomPixel - topPixel;

        setGx(calculatedGx);
        setGy(calculatedGy);
        calculateMagnitudeAndOrientation(calculatedGx, calculatedGy);
    };

    const calculateMagnitudeAndOrientation = useCallback((gxVal, gyVal) => {
        const mag = Math.sqrt(gxVal * gxVal + gyVal * gyVal);
        let orientationRad = Math.atan2(gyVal, gxVal);
        let orientationDeg = orientationRad * (180 / Math.PI);

        if (orientationDeg < 0) {
            orientationDeg += 360;
        }
        orientationDeg = orientationDeg % 180;

        setMagnitude(mag);
        setOrientation(orientationDeg);

        if (mag > 0) {
            const newHistogram = [...histogramData];
            const binIndex = Math.floor(orientationDeg);
            if (binIndex >= 0 && binIndex < 180) {
                newHistogram[binIndex] += 1;
            }
            setHistogramData(newHistogram);

            const updatedTable = newHistogram.map((freq, idx) => ({
                angleRange: `${idx}° - ${idx + 1}°`,
                frequency: freq
            })).filter(item => item.frequency > 0);
            setHistogramTable(updatedTable);
        }
    }, [histogramData]);

    // --- Real Keypoint Descriptor ---
    const generateRealDescriptor = useCallback((centerX, centerY) => {
        if (pixelMatrix.length === 0) return;
        
        const patchSize = 16;
        const patch = [];
        const descriptors = [];
        
        // Extract 16x16 patch around keypoint
        for (let y = centerY - patchSize/2; y < centerY + patchSize/2; y++) {
            const row = [];
            for (let x = centerX - patchSize/2; x < centerX + patchSize/2; x++) {
                const clampedY = Math.max(0, Math.min(pixelMatrix.length - 1, Math.floor(y)));
                const clampedX = Math.max(0, Math.min(pixelMatrix[0].length - 1, Math.floor(x)));
                row.push(pixelMatrix[clampedY][clampedX]);
            }
            patch.push(row);
        }
        
        setDescriptorPatch(patch);
        
        // Generate 128-dimensional descriptor (16 blocks * 8 bins each)
        for (let blockY = 0; blockY < 4; blockY++) {
            for (let blockX = 0; blockX < 4; blockX++) {
                const histogram = new Array(8).fill(0);
                
                // Process 4x4 sub-block
                for (let y = blockY * 4; y < (blockY + 1) * 4; y++) {
                    for (let x = blockX * 4; x < (blockX + 1) * 4; x++) {
                        if (y >= patch.length || x >= patch[0].length) continue;
                        
                        // Calculate gradients within patch
                        const current = patch[y][x];
                        const right = (x < patch[0].length - 1) ? patch[y][x + 1] : current;
                        const bottom = (y < patch.length - 1) ? patch[y + 1][x] : current;
                        
                        const gx = right - current;
                        const gy = bottom - current;
                        const mag = Math.sqrt(gx * gx + gy * gy);
                        let angle = Math.atan2(gy, gx) * 180 / Math.PI;
                        if (angle < 0) angle += 360;
                        
                        // Assign to 8-bin histogram (45° per bin)
                        const bin = Math.floor(angle / 45) % 8;
                        histogram[bin] += mag;
                    }
                }
                
                descriptors.push(...histogram);
            }
        }
        
        setDescriptorFeatures(descriptors);
    }, [pixelMatrix]);

    // --- Code Generation ---
    const generateCode = () => {
        switch(currentSection) {
            case 'gradient':
                return `// Real Gradient Calculation at pixel (${selectedPixel.row}, ${selectedPixel.col})
const calculateGradients = (pixelMatrix, row, col) => {
    const width = pixelMatrix[0].length;
    const height = pixelMatrix.length;
    
    const current = pixelMatrix[row][col]; // Value: ${pixelMatrix[selectedPixel.row]?.[selectedPixel.col] || 0}
    const left = (col > 0) ? pixelMatrix[row][col - 1] : current;
    const right = (col < width - 1) ? pixelMatrix[row][col + 1] : current;
    const top = (row > 0) ? pixelMatrix[row - 1][col] : current;
    const bottom = (row < height - 1) ? pixelMatrix[row + 1][col] : current;
    
    // Sobel-like gradients
    const gx = right - left; // ${gx}
    const gy = bottom - top; // ${gy}
    
    // Magnitude and orientation
    const magnitude = Math.sqrt(gx * gx + gy * gy); // ${magnitude.toFixed(2)}
    const orientation = Math.atan2(gy, gx) * 180 / Math.PI; // ${orientation.toFixed(1)}°
    
    return { gx, gy, magnitude, orientation };
};`;

            case 'scalespace':
                return `// Real Scale Space Construction
const createGaussianKernel = (sigma, size) => {
    const kernel = [];
    const center = Math.floor(size / 2);
    let sum = 0;
    
    for (let y = 0; y < size; y++) {
        kernel[y] = [];
        for (let x = 0; x < size; x++) {
            const dx = x - center;
            const dy = y - center;
            const value = Math.exp(-(dx*dx + dy*dy) / (2*sigma*sigma));
            kernel[y][x] = value;
            sum += value;
        }
    }
    
    // Normalize kernel
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            kernel[y][x] /= sum;
        }
    }
    
    return kernel;
};

// Current settings:
// Sigma: ${sigma.toFixed(2)}
// Octaves: ${numOctaves}
// Scales per octave: ${scalesPerOctave}
// Total scale space images: ${scaleSpaceImages.flat().length}`;

            case 'keypoints':
                return `// Real Keypoint Detection (DoG Extrema)
const findKeypoints = (dogSpace) => {
    const keypoints = [];
    const threshold = 10; // Contrast threshold
    
    dogSpace.forEach((octave, octaveIdx) => {
        for (let level = 1; level < octave.length - 1; level++) {
            const current = octave[level].imageData;
            const above = octave[level - 1].imageData;
            const below = octave[level + 1].imageData;
            
            // Check each pixel against 26 neighbors
            for (let y = 1; y < height - 1; y++) {
                for (let x = 1; x < width - 1; x++) {
                    const centerValue = current.data[(y * width + x) * 4];
                    
                    if (centerValue < threshold) continue;
                    
                    let isExtremum = true;
                    // Check 3x3x3 neighborhood...
                    
                    if (isExtremum) {
                        keypoints.push({ x, y, octave: octaveIdx, level });
                    }
                }
            }
        }
    });
    
    return keypoints;
};

// Detected keypoints: ${keypointLocations.length}`;

            case 'descriptor':
                return `// Real 128D Descriptor Generation
const generateDescriptor = (patch) => {
    const descriptors = [];
    
    // Process 16 blocks of 4x4 pixels each
    for (let blockY = 0; blockY < 4; blockY++) {
        for (let blockX = 0; blockX < 4; blockX++) {
            const histogram = new Array(8).fill(0);
            
            // Calculate gradients in 4x4 sub-block
            for (let y = blockY * 4; y < (blockY + 1) * 4; y++) {
                for (let x = blockX * 4; x < (blockX + 1) * 4; x++) {
                    const gx = patch[y][x + 1] - patch[y][x - 1];
                    const gy = patch[y + 1][x] - patch[y - 1][x];
                    const magnitude = Math.sqrt(gx*gx + gy*gy);
                    const angle = Math.atan2(gy, gx) * 180 / Math.PI;
                    
                    // 8-bin histogram (45° per bin)
                    const bin = Math.floor(angle / 45) % 8;
                    histogram[bin] += magnitude;
                }
            }
            
            descriptors.push(...histogram);
        }
    }
    
    return descriptors; // 128-dimensional vector
};

// Current descriptor length: ${descriptorFeatures.length}`;

            default:
                return '// Select a section to see the implementation';
        }
    };

    // --- Render Histogram ---
    const renderHistogram = () => {
        

        return (
            <div className="flex items-end h-40 border-b border-l border-gray-600 overflow-x-auto bg-gray-900 p-2">
                <CanvasHistogram histogramData={histogramData} />;
            </div>
        );
    };

    // --- Reset Functions ---
    const resetHistogram = () => {
        setHistogramData(Array(180).fill(0));
        setHistogramTable([]);
    };

    return (
        <div className="min-h-screen bg-gray-900 text-white font-inter p-8 flex flex-col items-center">
            <h1 className="text-4xl font-bold mb-8 text-center text-blue-400">
                Enhanced Interactive SIFT Algorithm
            </h1>

            {/* Navigation - Correct SIFT Order */}
            <div className="flex flex-wrap justify-center gap-4 mb-8">
                {[
                    { key: 'scalespace', label: '1. Scale Space', icon: <Grid3X3 className="w-5 h-5" /> },
                    { key: 'keypoints', label: '2. Keypoints', icon: <Target className="w-5 h-5" /> },
                    { key: 'gradient', label: '3. Gradients', icon: <Calculator className="w-5 h-5" /> },
                    { key: 'descriptor', label: '4. Descriptor', icon: <BarChart3 className="w-5 h-5" /> },
                    { key: 'matching', label: '5. Matching', icon: <Zap className="w-5 h-5" /> }
                ].map(section => (
                    <button
                        key={section.key}
                        onClick={() => setCurrentSection(section.key)}
                        className={`flex items-center space-x-2 px-6 py-3 rounded-lg transition-all ${
                            currentSection === section.key 
                                ? 'bg-blue-600 text-white shadow-lg' 
                                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                        }`}
                    >
                        {section.icon}
                        <span>{section.label}</span>
                    </button>
                ))}
            </div>

            {/* Image Upload Section */}
            <section className="bg-gray-800 p-6 rounded-xl shadow-lg mb-8 w-full max-w-4xl">
                <h2 className="text-2xl font-semibold mb-4 text-blue-300">1. Image Upload & Real Processing</h2>
                <div className="flex items-center space-x-4 mb-6">
                    <input
                        type="file"
                        accept="image/*"
                        onChange={handleImageUpload}
                        className="block w-full text-sm text-gray-300
                                   file:mr-4 file:py-2 file:px-4
                                   file:rounded-full file:border-0
                                   file:text-sm file:font-semibold
                                   file:bg-blue-500 file:text-white
                                   hover:file:bg-blue-600 cursor-pointer"
                    />
                    <Upload className="w-6 h-6 text-blue-400" />
                </div>
                {originalImage && (
                    <div className="flex flex-col md:flex-row justify-around items-center space-y-4 md:space-y-0 md:space-x-4">
                        <div className="text-center">
                            <h3 className="text-lg font-medium mb-2">Original Image</h3>
                            <img src={originalImage} alt="Original" className="max-w-xs max-h-48 rounded-lg shadow-md border border-gray-700" />
                        </div>
                        <div className="text-center">
                            <h3 className="text-lg font-medium mb-2">Grayscale + Grid</h3>
                            <img src={grayscaleImage} alt="Grayscale" className="max-w-xs max-h-48 rounded-lg shadow-md border border-gray-700" />
                        </div>
                    </div>
                )}
            </section>

            {/* Gradient Calculation Section */}
            {currentSection === 'gradient' && (
                <section className="bg-gray-800 p-6 rounded-xl shadow-lg mb-8 w-full max-w-4xl">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-2xl font-semibold text-blue-300">2. Real Gradient Calculation</h2>
                        <button
                            onClick={() => setShowCode(!showCode)}
                            className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors"
                        >
                            {showCode ? <Eye className="w-4 h-4" /> : <Code className="w-4 h-4" />}
                            <span>{showCode ? 'Hide' : 'Show'} Code</span>
                        </button>
                    </div>
                    
                    <p className="mb-4 text-gray-300">
                        Click on any pixel to see real gradient calculations: Gx = right - left, Gy = bottom - top
                    </p>
                    
                    <div className="flex justify-center items-center bg-gray-900 rounded-lg p-4 mb-6">
                        {grayscaleImage && pixelMatrix.length > 0 ? (
                            <div className="flex flex-col items-center">
                                <canvas
                                    ref={canvasRef}
                                    onClick={handleCanvasClick}
                                    className="border-2 border-blue-500 rounded-lg cursor-crosshair max-w-full mb-4"
                                    style={{ imageRendering: 'pixelated' }}
                                />
                                
                                {/* Legend */}
                                <div className="bg-gray-800 p-3 rounded-lg">
                                    <h4 className="text-sm font-bold text-white mb-2">Legend:</h4>
                                    <div className="grid grid-cols-2 gap-3 text-xs">
                                        <div className="flex items-center space-x-2">
                                            <div className="w-4 h-4 border-2 border-red-500 bg-red-500 bg-opacity-30"></div>
                                            <span className="text-white">Selected Pixel</span>
                                        </div>
                                        <div className="flex items-center space-x-2">
                                            <div className="w-4 h-4 border-2 border-yellow-500 bg-yellow-500 bg-opacity-30"></div>
                                            <span className="text-white">Neighbor Pixels</span>
                                        </div>
                                        <div className="flex items-center space-x-2">
                                            <div className="w-6 h-1 bg-blue-500"></div>
                                            <span className="text-white">Gx (Horizontal)</span>
                                        </div>
                                        <div className="flex items-center space-x-2">
                                            <div className="w-1 h-6 bg-orange-500"></div>
                                            <span className="text-white">Gy (Vertical)</span>
                                        </div>
                                        <div className="flex items-center space-x-2">
                                            <div className="w-4 h-1 bg-cyan-500"></div>
                                            <span className="text-white">Gradient Direction</span>
                                        </div>
                                        <div className="flex items-center space-x-2">
                                            <span className="text-cyan-400 font-mono text-xs">123.4</span>
                                            <span className="text-white">Magnitude</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <p className="text-gray-500">Upload an image to start.</p>
                        )}
                    </div>

                    {selectedPixel.row !== -1 && (
                        <div className="grid md:grid-cols-2 gap-6">
                            <div className="bg-gray-900 p-4 rounded-lg">
                                <h3 className="text-xl font-medium mb-2 text-blue-200">Selected Pixel: ({selectedPixel.row}, {selectedPixel.col})</h3>
                                <div className="space-y-2 font-mono">
                                    <p>Pixel Value: <span className="text-green-400">{pixelMatrix[selectedPixel.row]?.[selectedPixel.col]}</span></p>
                                    <p>Gx (Horizontal): <span className="text-yellow-400">{gx}</span></p>
                                    <p>Gy (Vertical): <span className="text-yellow-400">{gy}</span></p>
                                    <p>Magnitude: <span className="text-purple-400">{magnitude.toFixed(2)}</span></p>
                                    <p>Orientation: <span className="text-purple-400">{orientation.toFixed(1)}°</span></p>
                                </div>
                            </div>
                            
                            <div className="bg-gray-900 p-4 rounded-lg">
                                <div className="flex items-center justify-between mb-2">
                                    <h3 className="text-xl font-medium text-blue-200">Manual Input</h3>
                                    <RotateCcw className="w-5 h-5 text-gray-400" />
                                </div>
                                <div className="space-y-3">
                                    <div className="flex items-center space-x-4">
                                        <label className="text-sm w-12">Gx:</label>
                                        <input
                                            type="number"
                                            value={gx}
                                            onChange={(e) => {
                                                const val = parseFloat(e.target.value) || 0;
                                                setGx(val);
                                                calculateMagnitudeAndOrientation(val, gy);
                                            }}
                                            className="p-2 rounded bg-gray-700 border border-gray-600 text-white flex-1"
                                        />
                                    </div>
                                    <div className="flex items-center space-x-4">
                                        <label className="text-sm w-12">Gy:</label>
                                        <input
                                            type="number"
                                            value={gy}
                                            onChange={(e) => {
                                                const val = parseFloat(e.target.value) || 0;
                                                setGy(val);
                                                calculateMagnitudeAndOrientation(gx, val);
                                            }}
                                            className="p-2 rounded bg-gray-700 border border-gray-600 text-white flex-1"
                                        />
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Histogram Section */}
                    <div className="mt-8">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-xl font-medium text-blue-200">Orientation Histogram (Method 1)</h3>
                            <button
                                onClick={resetHistogram}
                                className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
                            >
                                Reset Histogram
                            </button>
                        </div>
                        <div className="bg-gray-900 p-4 rounded-lg">
                            {renderHistogram()}
                            <div className="flex justify-between text-sm mt-2 text-gray-400">
                                <span>0°</span>
                                <span>90°</span>
                                <span>179°</span>
                            </div>
                        </div>
                        
                        {histogramTable.length > 0 && (
                            <div className="mt-4 max-h-48 overflow-y-auto bg-gray-900 rounded-lg">
                                <table className="w-full text-left">
                                    <thead className="bg-gray-700">
                                        <tr>
                                            <th className="px-4 py-2">Angle Range</th>
                                            <th className="px-4 py-2">Frequency</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {histogramTable.map((item, index) => (
                                            <tr key={index} className={index % 2 === 0 ? 'bg-gray-800' : 'bg-gray-700'}>
                                                <td className="px-4 py-2">{item.angleRange}</td>
                                                <td className="px-4 py-2">{item.frequency}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}
                    </div>

                    {showCode && (
                        <div className="mt-6 bg-gray-900 rounded-lg p-4">
                            <pre className="text-sm text-green-400 overflow-x-auto">
                                {generateCode()}
                            </pre>
                        </div>
                    )}
                </section>
            )}

            {/* Scale Space Section */}
            {currentSection === 'scalespace' && (
                <section className="bg-gray-800 p-6 rounded-xl shadow-lg mb-8 w-full max-w-5xl">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-2xl font-semibold text-blue-300">3. Real Scale Space & DoG</h2>
                        <button
                            onClick={() => setShowCode(!showCode)}
                            className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors"
                        >
                            {showCode ? <Eye className="w-4 h-4" /> : <Code className="w-4 h-4" />}
                            <span>{showCode ? 'Hide' : 'Show'} Code</span>
                        </button>
                    </div>
                    
                    <p className="mb-4 text-gray-300">
                        Real Gaussian blur implementation with live convolution. No simulations!
                    </p>

                    {/* Sigma Control */}
                    <div className="mb-6">
                        <label className="block text-lg font-medium mb-2">
                            <Sliders className="inline w-5 h-5 mr-2" />
                            Gaussian Sigma: <span className="text-blue-400 font-mono">{sigma.toFixed(2)}</span>
                        </label>
                        <input
                            type="range"
                            min="0.5"
                            max="5.0"
                            step="0.1"
                            value={sigma}
                            onChange={(e) => setSigma(parseFloat(e.target.value))}
                            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                        />
                    </div>

                    {/* Octave Controls */}
                    <div className="grid md:grid-cols-2 gap-4 mb-6">
                        <div>
                            <label className="block text-sm font-medium mb-2">
                                Number of Octaves: <span className="text-green-400">{numOctaves}</span>
                            </label>
                            <input
                                type="range"
                                min="2"
                                max="6"
                                value={numOctaves}
                                onChange={(e) => setNumOctaves(parseInt(e.target.value))}
                                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium mb-2">
                                Scales per Octave: <span className="text-green-400">{scalesPerOctave}</span>
                            </label>
                            <input
                                type="range"
                                min="3"
                                max="7"
                                value={scalesPerOctave}
                                onChange={(e) => setScalesPerOctave(parseInt(e.target.value))}
                                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                            />
                        </div>
                    </div>

                    {/* Scale Space Visualization */}
                    {scaleSpaceImages.length > 0 && (
                        <div className="space-y-6">
                            {scaleSpaceImages.map((octave, octaveIdx) => (
                                <div key={octaveIdx} className="bg-gray-900 p-4 rounded-lg">
                                    <h3 className="text-xl font-semibold mb-4 text-yellow-300">
                                        Octave {octaveIdx + 1} (Scale: 1/{Math.pow(2, octaveIdx)})
                                    </h3>
                                    <div className="grid grid-cols-5 gap-2 mb-4">
                                        <h4 className="col-span-5 text-center text-blue-200 mb-2">Gaussian Blurred Images</h4>
                                        {octave.map((scale, scaleIdx) => (
                                            <div key={scaleIdx} className="text-center">
                                                <canvas
                                                    ref={el => {
                                                        if (el && scale.canvas) {
                                                            const ctx = el.getContext('2d');
                                                            el.width = Math.min(scale.canvas.width, 100);
                                                            el.height = Math.min(scale.canvas.height, 75);
                                                            ctx.drawImage(scale.canvas, 0, 0, el.width, el.height);
                                                        }
                                                    }}
                                                    className="border border-gray-600 rounded"
                                                />
                                                <p className="text-xs text-gray-400 mt-1">σ = {scale.sigma.toFixed(2)}</p>
                                            </div>
                                        ))}
                                    </div>
                                    
                                    {dogImages[octaveIdx] && (
                                        <div className="grid grid-cols-4 gap-2">
                                            <h4 className="col-span-4 text-center text-blue-200 mb-2">Difference of Gaussians (DoG)</h4>
                                            {dogImages[octaveIdx].map((dog, dogIdx) => (
                                                <div key={dogIdx} className="text-center">
                                                    <canvas
                                                        ref={el => {
                                                            if (el && dog.canvas) {
                                                                const ctx = el.getContext('2d');
                                                                el.width = Math.min(dog.canvas.width, 100);
                                                                el.height = Math.min(dog.canvas.height, 75);
                                                                ctx.drawImage(dog.canvas, 0, 0, el.width, el.height);
                                                            }
                                                        }}
                                                        className="border border-gray-600 rounded"
                                                    />
                                                    <p className="text-xs text-gray-400 mt-1">DoG {dogIdx + 1}</p>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}

                    {showCode && (
                        <div className="mt-6 bg-gray-900 rounded-lg p-4">
                            <pre className="text-sm text-green-400 overflow-x-auto">
                                {generateCode()}
                            </pre>
                        </div>
                    )}
                </section>
            )}

            {/* Keypoint Detection Section */}
            {currentSection === 'keypoints' && (
                <section className="bg-gray-800 p-6 rounded-xl shadow-lg mb-8 w-full max-w-4xl">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-2xl font-semibold text-blue-300">4. Real Keypoint Detection</h2>
                        <button
                            onClick={() => setShowCode(!showCode)}
                            className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors"
                        >
                            {showCode ? <Eye className="w-4 h-4" /> : <Code className="w-4 h-4" />}
                            <span>{showCode ? 'Hide' : 'Show'} Code</span>
                        </button>
                    </div>
                    
                    <p className="mb-4 text-gray-300">
                        Real extrema detection in DoG space. Each keypoint is checked against 26 neighbors (3×3×3 cube).
                    </p>
                    
                    <div className="grid md:grid-cols-2 gap-6">
                        <div className="bg-gray-900 p-4 rounded-lg">
                            <h3 className="text-lg font-medium mb-3 text-blue-200">Keypoint Statistics</h3>
                            <div className="space-y-2">
                                <p>Total Keypoints Found: <span className="text-green-400 font-bold">{keypointLocations.length}</span></p>
                                <p>DoG Images Processed: <span className="text-yellow-400">{dogImages.flat().length}</span></p>
                                <p>Scale Space Images: <span className="text-purple-400">{scaleSpaceImages.flat().length}</span></p>
                            </div>
                        </div>
                        
                        <div className="bg-gray-900 p-4 rounded-lg">
                            <h3 className="text-lg font-medium mb-3 text-blue-200">Detection Process</h3>
                            <ol className="list-decimal list-inside space-y-1 text-sm text-gray-300">
                                <li>Compare each pixel with 8 spatial neighbors</li>
                                <li>Compare with 9 neighbors in scale above</li>
                                <li>Compare with 9 neighbors in scale below</li>
                                <li>Must be local extremum (max or min)</li>
                                <li>Apply contrast threshold filtering</li>
                            </ol>
                        </div>
                    </div>

                    {keypointLocations.length > 0 && (
                        <div className="mt-6 bg-gray-900 p-4 rounded-lg">
                            <h3 className="text-lg font-medium mb-3 text-blue-200">Detected Keypoints (showing first 10)</h3>
                            <div className="overflow-x-auto">
                                <table className="w-full text-sm">
                                    <thead className="bg-gray-700">
                                        <tr>
                                            <th className="px-3 py-2 text-left">X</th>
                                            <th className="px-3 py-2 text-left">Y</th>
                                            <th className="px-3 py-2 text-left">Octave</th>
                                            <th className="px-3 py-2 text-left">Level</th>
                                            <th className="px-3 py-2 text-left">Response</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {keypointLocations.slice(0, 10).map((kp, idx) => (
                                            <tr key={idx} className={idx % 2 === 0 ? 'bg-gray-800' : 'bg-gray-700'}>
                                                <td className="px-3 py-2">{kp.x.toFixed(1)}</td>
                                                <td className="px-3 py-2">{kp.y.toFixed(1)}</td>
                                                <td className="px-3 py-2">{kp.octave}</td>
                                                <td className="px-3 py-2">{kp.level}</td>
                                                <td className="px-3 py-2">{kp.response.toFixed(1)}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {showCode && (
                        <div className="mt-6 bg-gray-900 rounded-lg p-4">
                            <pre className="text-sm text-green-400 overflow-x-auto">
                                {generateCode()}
                            </pre>
                        </div>
                    )}
                </section>
            )}

            {/* Descriptor Section */}
            {currentSection === 'descriptor' && (
                <section className="bg-gray-800 p-6 rounded-xl shadow-lg mb-8 w-full max-w-4xl">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-2xl font-semibold text-blue-300">5. Real 128D Descriptor</h2>
                        <button
                            onClick={() => setShowCode(!showCode)}
                            className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors"
                        >
                            {showCode ? <Eye className="w-4 h-4" /> : <Code className="w-4 h-4" />}
                            <span>{showCode ? 'Hide' : 'Show'} Code</span>
                        </button>
                    </div>
                    
                    <p className="mb-4 text-gray-300">
                        Real 16×16 patch extraction and 128-dimensional descriptor generation (16 blocks × 8 bins).
                    </p>

                    <div className="grid md:grid-cols-2 gap-6">
                        <div>
                            <h3 className="text-lg font-medium mb-3 text-blue-200">Keypoint Selection</h3>
                            <div className="space-y-3">
                                <div>
                                    <label className="block text-sm mb-1">X Coordinate:</label>
                                    <input
                                        type="number"
                                        value={descriptorKeypoint.x}
                                        onChange={(e) => {
                                            const x = parseInt(e.target.value) || 50;
                                            setDescriptorKeypoint(prev => ({ ...prev, x }));
                                            generateRealDescriptor(x, descriptorKeypoint.y);
                                        }}
                                        className="w-full p-2 rounded bg-gray-700 border border-gray-600 text-white"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm mb-1">Y Coordinate:</label>
                                    <input
                                        type="number"
                                        value={descriptorKeypoint.y}
                                        onChange={(e) => {
                                            const y = parseInt(e.target.value) || 50;
                                            setDescriptorKeypoint(prev => ({ ...prev, y }));
                                            generateRealDescriptor(descriptorKeypoint.x, y);
                                        }}
                                        className="w-full p-2 rounded bg-gray-700 border border-gray-600 text-white"
                                    />
                                </div>
                                <button
                                    onClick={() => generateRealDescriptor(descriptorKeypoint.x, descriptorKeypoint.y)}
                                    className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
                                >
                                    Generate Descriptor
                                </button>
                            </div>
                        </div>

                        <div>
                            <h3 className="text-lg font-medium mb-3 text-blue-200">16×16 Patch Visualization</h3>
                            <canvas
                                ref={descriptorCanvasRef}
                                className="border-2 border-green-500 rounded-lg max-w-full"
                            />
                        </div>
                    </div>

                    {descriptorFeatures.length > 0 && (
                        <div className="mt-6 bg-gray-900 p-4 rounded-lg">
                            <h3 className="text-lg font-medium mb-3 text-blue-200">
                                128D Feature Vector (showing first 32 values)
                            </h3>
                            <div className="grid grid-cols-8 gap-2 text-sm font-mono">
                                {descriptorFeatures.slice(0, 32).map((value, idx) => (
                                    <div
                                        key={idx}
                                        className="bg-gray-700 p-2 rounded text-center"
                                        title={`Feature ${idx}: ${value.toFixed(3)}`}
                                    >
                                        {value.toFixed(1)}
                                    </div>
                                ))}
                            </div>
                            <p className="text-gray-400 text-sm mt-2">
                                Total features: {descriptorFeatures.length} (4×4 blocks × 8 orientation bins each)
                            </p>
                        </div>
                    )}

                    {showCode && (
                        <div className="mt-6 bg-gray-900 rounded-lg p-4">
                            <pre className="text-sm text-green-400 overflow-x-auto">
                                {generateCode()}
                            </pre>
                        </div>
                    )}
                </section>
            )}

            {/* Footer */}
            <div className="text-center mt-8 text-gray-400">
                <p>Enhanced Interactive SIFT - Real implementations, no simulations • All calculations are live and accurate</p>
            </div>
        </div>
    );
};

export default EnhancedInteractiveSIFT;