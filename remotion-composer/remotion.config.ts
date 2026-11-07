import { Config } from "@remotion/cli/config";

// Use a locally installed Chrome/Chromium (headless shell) instead of letting
// Remotion download its own — needed on machines without access to
// remotion.media. Unset = Remotion's default behavior.
const browserExecutable = process.env.REMOTION_BROWSER_EXECUTABLE;
if (browserExecutable) {
  Config.setBrowserExecutable(browserExecutable);
}
