import React, { useRef, useEffect } from 'react';

const CanvasHistogram = ({ histogramData, width = 360, height = 160 }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    
    // Set canvas size accounting for device pixel ratio
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    const maxFreq = Math.max(...histogramData);
    if (maxFreq === 0) {
      // Draw "no data" message
      ctx.fillStyle = '#9CA3AF';
      ctx.font = '14px system-ui, -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Click pixels to build the histogram.', width / 2, height / 2);
      return;
    }

    // Draw histogram bars
    const barWidth = width / histogramData.length;
    const maxBarHeight = height - 20; // Leave space for padding

    histogramData.forEach((freq, index) => {
      const barHeight = Math.max((freq / maxFreq) * maxBarHeight, 1);
      const x = index * barWidth;
      const y = height - barHeight;

      // Draw bar
      ctx.fillStyle = '#3B82F6'; // blue-500
      ctx.fillRect(x, y, Math.max(barWidth - 0.5, 1), barHeight);
    });

    // Draw axes
    ctx.strokeStyle = '#4B5563'; // gray-600
    ctx.lineWidth = 1;
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(0, height);
    ctx.stroke();
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(0, height);
    ctx.lineTo(width, height);
    ctx.stroke();

  }, [histogramData, width, height]);

  // Handle mouse hover for tooltip
  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const barWidth = width / histogramData.length;
    const index = Math.floor(x / barWidth);
    
    if (index >= 0 && index < histogramData.length) {
      const freq = histogramData[index];
      canvas.title = `Angle: ${index}° | Frequency: ${freq}`;
    }
  };

  return (
    <div className="bg-gray-900 p-2 border-b border-l border-gray-600">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="block cursor-crosshair"
        style={{ width: `${width}px`, height: `${height}px` }}
        onMouseMove={handleMouseMove}
      />
    </div>
  );
};

export default CanvasHistogram;