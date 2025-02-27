            # Define thresholds
            distance_threshold = self.proximity
            angle_threshold = np.radians(30)  # 10-degree tolerance

            # If they are too close and have similar angles, discard the new line
            if distance_to_existing < distance_threshold and abs(angle_diff) < angle_threshold:
                existing_line_distance = self.EuclideanDistance(existing_line[0], existing_line[1])

                # Keep the longer line
                if new_line_distance > existing_line_distance:
                    self.lines.remove(existing_line)
                    self.lines.append([p1, p2])
                return False