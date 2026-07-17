* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: Arial, Helvetica, sans-serif;
  color: #1f2933;
  background: #f5f7fa;
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: center;
  padding: 16px 22px;
  background: #ffffff;
  border-bottom: 1px solid #dce3eb;
}

.topbar h1 {
  margin: 0;
  font-size: 22px;
}

.topbar p {
  margin: 5px 0 0;
  font-size: 12px;
  color: #697586;
  word-break: break-all;
}

.toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

select,
button,
input[type="number"] {
  height: 34px;
  border: 1px solid #cbd5df;
  background: #ffffff;
  border-radius: 6px;
  padding: 0 10px;
  font-size: 13px;
}

button:disabled {
  opacity: 0.45;
}

.thresholdControl {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 34px;
  color: #4b5563;
  font-size: 12px;
  white-space: nowrap;
}

.thresholdControl input {
  width: 84px;
}

.pageInfo {
  min-width: 80px;
  color: #697586;
  font-size: 12px;
  white-space: nowrap;
}

.summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 10px;
  padding: 16px 22px;
}

.summary div {
  background: #ffffff;
  border: 1px solid #dfe6ee;
  border-radius: 8px;
  padding: 12px;
}

.summary b {
  display: block;
  font-size: 12px;
  color: #697586;
}

.summary span {
  display: block;
  margin-top: 4px;
  font-size: 22px;
  font-weight: 700;
}

.content {
  padding: 0 22px 22px;
}

.card {
  margin-bottom: 18px;
  padding: 16px;
  background: #ffffff;
  border: 1px solid #dfe6ee;
  border-radius: 8px;
}

.cardHead,
.blockHead {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: flex-start;
}

.cardHead {
  padding-bottom: 12px;
  border-bottom: 1px solid #edf1f5;
}

.cardHead h2,
.blockHead h3 {
  margin: 0;
}

.cardHead h2 {
  font-size: 18px;
}

.cardHead p {
  margin: 5px 0 0;
  color: #697586;
  font-size: 13px;
}

.flagBox,
.blockHead {
  display: flex;
  gap: 8px;
  align-items: center;
}

.pill {
  display: inline-flex;
  align-items: center;
  height: 24px;
  border-radius: 999px;
  padding: 0 9px;
  font-size: 12px;
  font-weight: 700;
}

.pill.ok {
  color: #116b3a;
  background: #e4f6ed;
}

.pill.bad {
  color: #9f2f2f;
  background: #fdeaea;
}

.target {
  display: grid;
  grid-template-columns: 220px minmax(280px, 1fr);
  gap: 12px;
  margin-top: 14px;
}

.poolLayout {
  display: grid;
  grid-template-columns: 180px 220px minmax(260px, 1fr);
  gap: 12px;
  margin-top: 14px;
}

.block {
  margin-top: 16px;
}

.blockHead {
  justify-content: flex-start;
  margin-bottom: 8px;
}

.blockHead h3 {
  font-size: 15px;
}

.reason {
  font-size: 12px;
  color: #697586;
}

.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 10px;
}

.thumbPair {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
}

.thumbWrap {
  margin: 0;
}

.thumbWrap img {
  cursor: zoom-in;
}

.thumbWrap figcaption {
  margin-top: 3px;
  text-align: center;
  font-size: 10px;
  color: #697586;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.tile,
.videoTile {
  min-width: 0;
  padding: 8px;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  background: #fbfcfe;
}

.thumb {
  width: 100%;
  height: 150px;
  object-fit: contain;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #7b8794;
  background: #edf2f7;
  border-radius: 4px;
}

.tile b,
.videoTile b {
  display: block;
  margin-top: 7px;
  font-size: 12px;
}

.videoPlaceholder {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  min-height: 150px;
  margin-top: 8px;
  padding: 14px;
  color: #697586;
  background: #edf2f7;
  border-radius: 4px;
}

.videoPlaceholder a {
  color: #1d4ed8;
  font-size: 13px;
  font-weight: 700;
  text-decoration: none;
}

.meta,
.stats {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 7px;
}

.meta span,
.stats span {
  padding: 3px 5px;
  border-radius: 4px;
  background: #edf2f7;
  color: #4b5563;
  font-size: 11px;
}

.path {
  margin-top: 6px;
  color: #697586;
  font-size: 11px;
  word-break: break-all;
}

.empty,
.emptyPage {
  padding: 16px;
  border: 1px dashed #cbd5df;
  border-radius: 6px;
  color: #697586;
  background: #fbfcfe;
}

.poolPanel {
  padding: 16px;
  background: #ffffff;
  border: 1px solid #dfe6ee;
  border-radius: 8px;
}

.poolPanelHead {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.poolPanelHead h2 {
  margin: 0;
  font-size: 18px;
}

.poolPanelHead span {
  color: #697586;
  font-size: 13px;
}

.poolGrid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.poolPhoto {
  width: 112px;
  padding: 6px;
  border: 1px solid #e2e8f0;
  border-radius: 5px;
  background: #ffffff;
}

.poolImgWrap {
  position: relative;
  width: 100px;
  height: 100px;
  overflow: hidden;
  border-radius: 4px;
  background: #edf2f7;
}

.poolImgWrap img {
  display: block;
  width: 100px;
  height: 100px;
  object-fit: cover;
  cursor: pointer;
  transition: transform 0.15s ease;
}

.poolImgWrap img:hover {
  transform: scale(1.05);
}

.poolPose {
  display: flex;
  justify-content: space-between;
  gap: 2px;
  margin-top: 5px;
  color: #303133;
  font-size: 9px;
}

.poolCaption {
  margin-top: 4px;
  color: #697586;
  font-size: 10px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.emotionBadge {
  position: absolute;
  left: 4px;
  bottom: 4px;
  padding: 2px 5px;
  border-radius: 999px;
  color: #ffffff;
  background: rgba(37, 99, 235, 0.9);
  font-size: 10px;
  font-weight: 700;
}

.emoFlag {
  position: absolute;
  right: 4px;
  top: 4px;
  padding: 2px 5px;
  border-radius: 999px;
  color: #ffffff;
  font-size: 9px;
  font-weight: 700;
}

.emoFlag.true {
  background: rgba(22, 163, 74, 0.9);
}

.emoFlag.false {
  background: rgba(220, 38, 38, 0.9);
}

.lightbox {
  display: none;
  position: fixed;
  inset: 0;
  z-index: 200;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.86);
  cursor: pointer;
}

.lightbox.active {
  display: flex;
}

.lightbox img {
  max-width: 90vw;
  max-height: 90vh;
  border-radius: 4px;
}

@media (max-width: 760px) {
  .topbar,
  .toolbar,
  .cardHead {
    align-items: stretch;
    flex-direction: column;
  }

  .target {
    grid-template-columns: 1fr;
  }

  .poolLayout {
    grid-template-columns: 1fr;
  }

  .content,
  .summary {
    padding-left: 10px;
    padding-right: 10px;
  }
}

.qualityBar {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 6px;
}

.qualityBar.detail {
  align-items: flex-start;
}

.qualityPill {
  display: inline-flex;
  align-items: center;
  max-width: 100%;
  padding: 2px 5px;
  border-radius: 999px;
  font-size: 10px;
  line-height: 1.2;
  border: 1px solid #cbd5e1;
  background: #f8fafc;
  color: #334155;
  overflow-wrap: anywhere;
}

.qualityPill.ok {
  border-color: #86efac;
  background: #ecfdf3;
  color: #166534;
}

.qualityPill.bad {
  border-color: #fca5a5;
  background: #fef2f2;
  color: #991b1b;
}

.qualityPill.neutral {
  border-color: #cbd5e1;
  background: #f8fafc;
  color: #64748b;
}
