import React, { Suspense, lazy } from "react";

import UploadPage from "./UploadPage";

// The only route in the application, and the reason there is no router: a
// share link (Phase 10) has to be a URL somebody can paste, and everything
// else is one page. `react-router` for a single conditional would be a
// dependency in the initial bundle to express one `if`.
const SharedRunPage = lazy(() => import("./SharedRunPage"));

const SHARE_PATH = /^\/share\/([A-Za-z0-9_-]+)\/?$/;

export default function App() {
  const match = SHARE_PATH.exec(window.location.pathname);
  if (match) {
    return (
      <Suspense fallback={null}>
        <SharedRunPage token={match[1]} />
      </Suspense>
    );
  }
  return <UploadPage />;
}
