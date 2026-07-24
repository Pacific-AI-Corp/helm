import HomeLanding from "@/components/Landing/HomeLanding";
import MedHELMLanding from "@/components/Landing/MedHELMLanding";
import MedHELMV1Landing from "@/components/Landing/MedHELMV1Landing";

export default function Home() {
  if (window.PROJECT_ID === "medhelm" && window.RELEASE == "v1.0.0") {
    return <MedHELMV1Landing />;
  } else if (window.PROJECT_ID === "medhelm") {
    return <MedHELMLanding />;
  } else if (window.PROJECT_ID === "home") {
    return <HomeLanding />;
  } else {
    return <MedHELMLanding />;
  }
}
