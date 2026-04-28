/**
 * Figma Plugin: LaunchVid Exporter
 * 
 * This plugin exports Figma frames with full layer trees, PNGs, and SVGs
 * for consumption by the LaunchVid backend (FastAPI pipeline).
 * 
 * Key features:
 * - Exports layer trees with positions, sizes, fills, text, and effects
 * - Pre-renders complex nodes (VECTOR, BOOLEAN_OPERATION, ELLIPSE, GROUP) as PNG/SVG
 * - Handles nested layers (children)
 * - Base64 encodes images for easy transmission
 */

// FIX 5 applied: Pre-render complex nodes as PNG/SVG

interface LayerExport {
  id: string;
  name: string;
  type: string;
  x: number;
  y: number;
  width: number;
  height: number;
  visible: boolean;
  opacity: number;
  cornerRadius: number;
  fill: any;
  stroke: any;
  text: any;
  children: LayerExport[];
  exportedImageBase64: string | null;
  exportedSvg: string | null;
}

interface FrameExport {
  frameId: string;
  frameName: string;
  width: number;
  height: number;
  fullPngBase64: string;
  layers: LayerExport;
  bgColor: string;
}

async function serializeNode(node: any): Promise<LayerExport> {
  const base: LayerExport = {
    id: node.id,
    name: node.name,
    type: node.type,
    x: node.x,
    y: node.y,
    width: node.width,
    height: node.height,
    visible: node.visible,
    opacity: node.opacity,
    cornerRadius: node.cornerRadius ?? 0,
    fill: extractFill(node),
    stroke: extractStroke(node),
    text: extractText(node),
    children: [],
    exportedImageBase64: null,
    exportedSvg: null,
  };

  // FIX 5: Always export VECTOR as SVG
  if (node.type === "VECTOR") {
    try {
      const svgData = await node.exportAsync({ format: "SVG" });
      const svgString = new TextDecoder().decode(svgData);
      base.exportedSvg = svgString;
      console.log(`[plugin] Exported VECTOR '${node.name}' as SVG`);
    } catch (e) {
      console.warn(`[plugin] Failed to export VECTOR '${node.name}' as SVG:`, e);
      // Fallback to PNG
      try {
        const pngData = await node.exportAsync({ format: "PNG", constraint: { type: "SCALE", value: 2 } });
        base.exportedImageBase64 = figma.base64Encode(pngData);
        console.log(`[plugin] Exported VECTOR '${node.name}' as PNG (fallback)`);
      } catch (e2) {
        console.warn(`[plugin] Failed to export VECTOR '${node.name}' as PNG:`, e2);
      }
    }
  }

  // FIX 5: Always export BOOLEAN_OPERATION as PNG
  else if (node.type === "BOOLEAN_OPERATION") {
    try {
      const pngData = await node.exportAsync({ format: "PNG", constraint: { type: "SCALE", value: 2 } });
      base.exportedImageBase64 = figma.base64Encode(pngData);
      console.log(`[plugin] Exported BOOLEAN_OPERATION '${node.name}' as PNG`);
    } catch (e) {
      console.warn(`[plugin] Failed to export BOOLEAN_OPERATION '${node.name}':`, e);
    }
  }

  // FIX 5: ELLIPSE with IMAGE or GRADIENT fill → always export as PNG
  else if (node.type === "ELLIPSE" && node.fills && node.fills.length > 0) {
    const hasFancyFill = node.fills.some((f: any) => f.type === "IMAGE" || f.type === "GRADIENT");
    if (hasFancyFill) {
      try {
        const pngData = await node.exportAsync({ format: "PNG", constraint: { type: "SCALE", value: 2 } });
        base.exportedImageBase64 = figma.base64Encode(pngData);
        console.log(`[plugin] Exported ELLIPSE '${node.name}' (with fancy fill) as PNG`);
      } catch (e) {
        console.warn(`[plugin] Failed to export ELLIPSE '${node.name}':`, e);
      }
    }
  }

  // FIX 5: GROUP nodes (complex layouts) → always export as PNG
  else if (node.type === "GROUP") {
    try {
      const pngData = await node.exportAsync({ format: "PNG", constraint: { type: "SCALE", value: 2 } });
      base.exportedImageBase64 = figma.base64Encode(pngData);
      console.log(`[plugin] Exported GROUP '${node.name}' as PNG`);
    } catch (e) {
      console.warn(`[plugin] Failed to export GROUP '${node.name}':`, e);
    }
  }

  // FIX 5: For IMAGE fills in any layer, export the referenced image
  else if (node.fills && node.fills.length > 0) {
    const imageFill = node.fills.find((f: any) => f.type === "IMAGE");
    if (imageFill && imageFill.imageHash) {
      try {
        const imageData = await figma.getImageByHash(imageFill.imageHash).getBytesAsync();
        base.exportedImageBase64 = figma.base64Encode(imageData);
      } catch (e) {
        console.warn(`[plugin] Failed to export image for '${node.name}':`, e);
      }
    }
  }

  // Recursively serialize children
  if (node.children) {
    for (const child of node.children) {
      if (child.visible !== false) {
        base.children.push(await serializeNode(child));
      }
    }
  }

  return base;
}

function extractFill(node: any): any {
  if (!node.fills || node.fills.length === 0) {
    return null;
  }

  const fill = node.fills[0];

  if (fill.type === "SOLID") {
    return {
      type: "SOLID",
      hex: rgbToHex(fill.color),
      opacity: fill.opacity,
    };
  } else if (fill.type === "GRADIENT_LINEAR" || fill.type === "GRADIENT_RADIAL") {
    return {
      type: "GRADIENT",
      gradientType: fill.type,
      gradientStops: fill.gradientStops.map((s: any) => ({
        hex: rgbToHex(s.color),
        position: s.position,
        opacity: s.color.a,
      })),
    };
  }

  return { type: fill.type };
}

function extractStroke(node: any): any {
  if (!node.strokes || node.strokes.length === 0) {
    return null;
  }

  const stroke = node.strokes[0];
  return {
    hex: rgbToHex(stroke.color),
    weight: node.strokeWeight,
  };
}

function extractText(node: any): any {
  if (node.type !== "TEXT" || !node.characters) {
    return null;
  }

  const textStyle = node.getStyledTextSegments(["fontFamily", "fontSize", "fontWeight", "textAlignHorizontal", "lineHeight", "letterSpacing", "fills"]);
  const firstSegment = textStyle[0] || {};

  return {
    characters: node.characters,
    fontSize: firstSegment.fontSize ?? node.fontSize ?? 16,
    fontWeight: firstSegment.fontWeight ?? node.fontWeight ?? 400,
    fontFamily: firstSegment.fontFamily ?? node.fontFamily ?? "Inter",
    color: extractTextColor(firstSegment.fills || node.fills),
    textAlign: firstSegment.textAlignHorizontal ?? node.textAlignHorizontal ?? "LEFT",
    lineHeight: node.lineHeight?.value ?? "AUTO",
    letterSpacing: node.letterSpacing?.value ?? 0,
  };
}

function extractTextColor(fills: any[]): string {
  if (!fills || fills.length === 0) {
    return "#000000";
  }

  const fill = fills[0];
  if (fill && fill.color) {
    return rgbToHex(fill.color);
  }

  return "#000000";
}

function rgbToHex(color: any): string {
  if (!color) return "#000000";
  const r = Math.round((color.r ?? 0) * 255);
  const g = Math.round((color.g ?? 0) * 255);
  const b = Math.round((color.b ?? 0) * 255);
  return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, "0")}`;
}

async function exportSelectedFrames(): Promise<FrameExport[]> {
  const selection = figma.currentPage.selection.filter((n: any) => n.type === "FRAME");

  if (selection.length === 0) {
    figma.notify("Please select at least one frame to export.", { error: true });
    return [];
  }

  const exports: FrameExport[] = [];

  for (const frame of selection) {
    console.log(`[plugin] Exporting frame: ${frame.name}`);

    // Export full frame as PNG for fallback
    let fullPngBase64 = "";
    try {
      const pngData = await frame.exportAsync({ format: "PNG" });
      fullPngBase64 = figma.base64Encode(pngData);
    } catch (e) {
      console.warn(`[plugin] Failed to export frame PNG for '${frame.name}':`, e);
    }

    // Serialize layer tree
    const layerTree = await serializeNode(frame);

    exports.push({
      frameId: frame.id,
      frameName: frame.name,
      width: frame.width,
      height: frame.height,
      fullPngBase64,
      layers: layerTree,
      bgColor: extractBgColor(frame),
    });
  }

  return exports;
}

function extractBgColor(frame: any): string {
  if (frame.fills && frame.fills.length > 0) {
    const fill = frame.fills[0];
    if (fill.type === "SOLID" && fill.color) {
      return rgbToHex(fill.color);
    }
  }
  return "#ffffff";
}

// Main plugin entry point
figma.showUI(__html__, { width: 400, height: 500 });

figma.ui.onmessage = async (msg: any) => {
  if (msg.type === "export") {
    const frames = await exportSelectedFrames();
    figma.ui.postMessage({ type: "export-done", frames });
  } else if (msg.type === "close") {
    figma.closePlugin();
  }
};
