import { Composition } from "remotion";
import { LaunchVideo } from "./Composition";

export const RemotionRoot = () => {
  return (
    <>
      <Composition
        id="LaunchVideo"
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
    </>
  );
};