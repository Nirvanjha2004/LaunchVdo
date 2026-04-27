import {
  AbsoluteFill,
  Audio,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
  Sequence,
  staticFile,
} from "remotion";

// ─── Types (mirrors what renderer.py sends) ────────────────────────────────────

interface AnimationStep {
  layer_id: string;
  layer_name: string;
  layer_type: string;
  animation: string;
  delay_ms: number;
  duration_ms: number;
  easing: string;
}

interface LayerData {
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
  fill: {
    type: string;
    hex?: string;
    opacity?: number;
    gradientType?: string;
    gradientStops?: { hex: string; position: number; opacity: number }[];
  } | null;
  stroke: { hex: string; weight: number } | null;
  text: {
    characters: string;
    fontSize: number;
    fontWeight: number;
    fontFamily: string;
    color: string;
    textAlign: string;
    lineHeight: number | "AUTO";
    letterSpacing: number;
  } | null;
  children: LayerData[];
  exportedImageBase64: string | null;
  exportedSvg: string | null;
}

interface FrameData {
  frameId: string;
  frameName: string;
  width: number;
  height: number;
  fullPngBase64: string;
  layers: LayerData;
  bgColor: string;
  animationSequence: AnimationStep[];
}

interface SceneData {
  sceneIndex: number;
  sceneType: "intro_card" | "screen" | "outro_card";
  startFrame: number;
  durationFrames: number;
  transitionIn: string;
  transitionOut: string;
  screenIndex: number | null;
  narration: string;
  audioPath: string | null;
}

export interface LaunchVideoProps {
  jobId: string;
  appName: string;
  tagline: string;
  totalFrames: number;
  fps: number;
  scenes: SceneData[];
  frames: FrameData[];
}

// ─── Helpers ───────────────────────────────────────────────────────────────────

function msToFrames(ms: number, fps: number): number {
  return Math.round((ms / 1000) * fps);
}

function getTransitionStyle(
  type: string,
  progress: number
): React.CSSProperties {
  switch (type) {
    case "slide_up":
      return { transform: `translateY(${interpolate(progress, [0, 1], [80, 0])}px)`, opacity: progress };
    case "slide_right":
      return { transform: `translateX(${interpolate(progress, [0, 1], [-80, 0])}px)`, opacity: progress };
    case "slide_left":
      return { transform: `translateX(${interpolate(progress, [0, 1], [80, 0])}px)`, opacity: progress };
    case "zoom_in":
      return { transform: `scale(${interpolate(progress, [0, 1], [0.85, 1])})`, opacity: progress };
    case "scale_in":
      return { transform: `scale(${interpolate(progress, [0, 1], [0.7, 1])})`, opacity: progress };
    case "fade":
    default:
      return { opacity: progress };
  }
}

// ─── Single animated layer ─────────────────────────────────────────────────────

const AnimatedLayer: React.FC<{
  layer: LayerData;
  animStep: AnimationStep | undefined;
  fps: number;
  frameOffset: number; // frame within this scene
  scaleX: number;
  scaleY: number;
}> = ({ layer, animStep, fps, frameOffset, scaleX, scaleY }) => {
  const delayFrames = animStep ? msToFrames(animStep.delay_ms, fps) : 0;
  const durationFrames = animStep ? msToFrames(animStep.duration_ms, fps) : 20;
  const localFrame = Math.max(0, frameOffset - delayFrames);

  const progress = spring({
    frame: localFrame,
    fps,
    durationInFrames: durationFrames,
    config: { damping: 14, stiffness: 180 },
  });

  const animationType = animStep?.animation ?? "fade_in";
  const transitionStyle = getTransitionStyle(animationType, progress);

  // pulse effect for CTA buttons
  const pulseScale = animationType === "pulse"
    ? 1 + 0.04 * Math.sin((frameOffset - delayFrames) * 0.3)
    : 1;

  const positionStyle: React.CSSProperties = {
    position: "absolute",
    left:   layer.x * scaleX,
    top:    layer.y * scaleY,
    width:  layer.width * scaleX,
    height: layer.height * scaleY,
    opacity: (layer.opacity ?? 1) * (transitionStyle.opacity ?? 1),
    borderRadius: layer.cornerRadius * Math.min(scaleX, scaleY),
    transform: `${transitionStyle.transform ?? ""} scale(${pulseScale})`,
    overflow: "hidden",
  };

  // ── Fill background ──
  let background: string | undefined;
  if (layer.fill?.type === "SOLID") {
    background = layer.fill.hex;
  } else if (layer.fill?.type === "GRADIENT" && layer.fill.gradientStops) {
    const stops = layer.fill.gradientStops
      .map(s => `${s.hex} ${Math.round(s.position * 100)}%`)
      .join(", ");
    background = `linear-gradient(135deg, ${stops})`;
  }

  // ── Stroke ──
  const outline = layer.stroke
    ? `${layer.stroke.weight}px solid ${layer.stroke.hex}`
    : undefined;

  // ── Content ──
  let content: React.ReactNode = null;

  if (layer.exportedSvg) {
    // Vector node exported as SVG
    content = (
      <div
        style={{ width: "100%", height: "100%" }}
        dangerouslySetInnerHTML={{ __html: layer.exportedSvg }}
      />
    );
  } else if (layer.exportedImageBase64) {
    // Image fill or rasterized vector/boolean op
    content = (
      <img
        src={`data:image/png;base64,${layer.exportedImageBase64}`}
        style={{ width: "100%", height: "100%", objectFit: "cover" }}
      />
    );
  } else if (layer.text) {
    const t = layer.text;
    const lh = t.lineHeight === "AUTO" ? "1.2" : `${t.lineHeight * scaleY}px`;
    content = (
      <div
        style={{
          color:          t.color,
          fontSize:       t.fontSize * Math.min(scaleX, scaleY),
          fontWeight:     t.fontWeight,
          fontFamily:     `"${t.fontFamily}", Inter, sans-serif`,
          textAlign:      t.textAlign.toLowerCase() as React.CSSProperties["textAlign"],
          lineHeight:     lh,
          letterSpacing:  t.letterSpacing,
          whiteSpace:     "pre-wrap",
          width:          "100%",
          height:         "100%",
          overflow:       "hidden",
        }}
      >
        {t.characters}
      </div>
    );
  }

  return (
    <div
      style={{
        ...positionStyle,
        background,
        outline,
      }}
    >
      {content}
    </div>
  );
};

// ─── Full screen scene ─────────────────────────────────────────────────────────

const ScreenScene: React.FC<{
  frameData: FrameData;
  scene: SceneData;
  fps: number;
  localFrame: number; // frame within this scene (0 = scene start)
}> = ({ frameData, scene, fps, localFrame }) => {
  const CANVAS_W = 1080;
  const CANVAS_H = 1920;

  // Scale the design to fit our canvas
  const scaleX = CANVAS_W / frameData.width;
  const scaleY = CANVAS_H / frameData.height;

  // Scene entry transition
  const entryProgress = spring({
    frame: localFrame,
    fps,
    durationInFrames: 18,
    config: { damping: 16 },
  });
  const sceneStyle = getTransitionStyle(scene.transitionIn, entryProgress);

  // Build animation map: layerId → AnimationStep
  const animMap = new Map<string, AnimationStep>(
    (frameData.animationSequence ?? []).map(a => [a.layer_id, a])
  );

  const layers = frameData.layers?.children ?? [];

  return (
    <AbsoluteFill
      style={{
        backgroundColor: frameData.bgColor ?? "#ffffff",
        ...sceneStyle,
      }}
    >
      {/* Fallback: full frame PNG underneath (in case layer rendering misses something) */}
      {frameData.fullPngBase64 && (
        <img
          src={`data:image/png;base64,${frameData.fullPngBase64}`}
          style={{
            position: "absolute",
            width: "100%",
            height: "100%",
            objectFit: "cover",
            opacity: 0.15, // very faint — let layer renders take priority
          }}
        />
      )}

      {/* Render each layer individually with its animation */}
      {layers.map(layer => (
        <AnimatedLayer
          key={layer.id}
          layer={layer}
          animStep={animMap.get(layer.id)}
          fps={fps}
          frameOffset={localFrame}
          scaleX={scaleX}
          scaleY={scaleY}
        />
      ))}
    </AbsoluteFill>
  );
};

// ─── Intro card ────────────────────────────────────────────────────────────────

const IntroCard: React.FC<{
  appName: string;
  tagline: string;
  localFrame: number;
  fps: number;
}> = ({ appName, tagline, localFrame, fps }) => {
  const titleProgress = spring({ frame: localFrame, fps, durationInFrames: 20, config: { damping: 14 } });
  const taglineProgress = spring({ frame: Math.max(0, localFrame - 12), fps, durationInFrames: 20, config: { damping: 14 } });

  return (
    <AbsoluteFill
      style={{
        background: "linear-gradient(160deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)",
        alignItems: "center",
        justifyContent: "center",
        flexDirection: "column",
        gap: 20,
      }}
    >
      <div
        style={{
          color:      "#ffffff",
          fontSize:   96,
          fontWeight: 800,
          fontFamily: "Inter, sans-serif",
          textAlign:  "center",
          padding:    "0 60px",
          opacity:    titleProgress,
          transform:  `translateY(${interpolate(titleProgress, [0, 1], [40, 0])}px)`,
        }}
      >
        {appName}
      </div>
      <div
        style={{
          color:      "rgba(255,255,255,0.7)",
          fontSize:   48,
          fontWeight: 400,
          fontFamily: "Inter, sans-serif",
          textAlign:  "center",
          padding:    "0 80px",
          opacity:    taglineProgress,
          transform:  `translateY(${interpolate(taglineProgress, [0, 1], [20, 0])}px)`,
        }}
      >
        {tagline}
      </div>
    </AbsoluteFill>
  );
};

// ─── Outro card ────────────────────────────────────────────────────────────────

const OutroCard: React.FC<{
  appName: string;
  localFrame: number;
  fps: number;
}> = ({ appName, localFrame, fps }) => {
  const progress = spring({ frame: localFrame, fps, durationInFrames: 20, config: { damping: 14 } });

  return (
    <AbsoluteFill
      style={{
        background:     "linear-gradient(160deg, #0f3460 0%, #16213e 100%)",
        alignItems:     "center",
        justifyContent: "center",
        flexDirection:  "column",
        gap:            32,
      }}
    >
      <div
        style={{
          color:       "#ffffff",
          fontSize:    80,
          fontWeight:  800,
          fontFamily:  "Inter, sans-serif",
          opacity:     progress,
          transform:   `scale(${interpolate(progress, [0, 1], [0.85, 1])})`,
        }}
      >
        {appName}
      </div>
      <div
        style={{
          background:   "#ffffff",
          color:        "#0f3460",
          fontSize:     44,
          fontWeight:   700,
          fontFamily:   "Inter, sans-serif",
          padding:      "20px 60px",
          borderRadius: 60,
          opacity:      progress,
          transform:    `translateY(${interpolate(progress, [0, 1], [30, 0])}px)`,
        }}
      >
        Download Now
      </div>
    </AbsoluteFill>
  );
};

// ─── Main composition ──────────────────────────────────────────────────────────

export const LaunchVideo: React.FC<LaunchVideoProps> = ({
  appName,
  tagline,
  fps,
  scenes,
  frames,
}) => {
  const { frame } = useVideoConfig();
  const currentFrame = useCurrentFrame();

  return (
    <AbsoluteFill style={{ backgroundColor: "#000000" }}>
      {scenes.map(scene => {
        const localFrame = currentFrame - scene.startFrame;

        return (
          <Sequence
            key={scene.sceneIndex}
            from={scene.startFrame}
            durationInFrames={scene.durationFrames}
          >
            {/* Voiceover audio */}
            {scene.audioPath && (
              <Audio src={staticFile(scene.audioPath)} />
            )}

            {/* Scene content */}
            {scene.sceneType === "intro_card" && (
              <IntroCard
                appName={appName}
                tagline={tagline}
                localFrame={localFrame}
                fps={fps}
              />
            )}

            {scene.sceneType === "screen" &&
              scene.screenIndex !== null &&
              frames[scene.screenIndex] && (
                <ScreenScene
                  frameData={frames[scene.screenIndex]}
                  scene={scene}
                  fps={fps}
                  localFrame={localFrame}
                />
              )}

            {scene.sceneType === "outro_card" && (
              <OutroCard
                appName={appName}
                localFrame={localFrame}
                fps={fps}
              />
            )}
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};