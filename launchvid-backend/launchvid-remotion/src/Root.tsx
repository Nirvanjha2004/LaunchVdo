import { Composition } from "remotion";
import { LaunchVideo } from "./Composition";

export const RemotionRoot = () => {
  return (
    <>
      {/* FIX 9 applied: Three output formats for different platforms */}

      {/* Format 1: Portrait (9:16) — App Store / Primary */}
      <Composition
        id="LaunchVideoPortrait"
        component={LaunchVideo}
        durationInFrames={900}   // 30 seconds @ 30fps
        fps={30}
        width={1080}
        height={1920}            // 9:16 portrait (App Store format)
        defaultProps={{
          jobId: "preview",
          appName: "My App",
          tagline: "Built for you.",
          totalFrames: 900,
          fps: 30,
          scenes: [],
          frames: [],
        }}
      />

      {/* Format 2: Landscape (16:9) — Twitter/X, YouTube Shorts */}
      <Composition
        id="LaunchVideoLandscape"
        component={LaunchVideo}
        durationInFrames={900}
        fps={30}
        width={1920}
        height={1080}            // 16:9 landscape
        defaultProps={{
          jobId: "preview",
          appName: "My App",
          tagline: "Built for you.",
          totalFrames: 900,
          fps: 30,
          scenes: [],
          frames: [],
        }}
      />

      {/* Format 3: Square (1:1) — Instagram, TikTok, Threads */}
      <Composition
        id="LaunchVideoSquare"
        component={LaunchVideo}
        durationInFrames={900}
        fps={30}
        width={1080}
        height={1080}            // 1:1 square
        defaultProps={{
          jobId: "preview",
          appName: "My App",
          tagline: "Built for you.",
          totalFrames: 900,
          fps: 30,
          scenes: [],
          frames: [],
        }}
      />
    </>
  );
};